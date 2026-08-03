// ─────────────────────────────────────────────────────────────────────────────
// PriceRef infrastructure — Azure Container Apps.
//
// Topology:
//
//   Internet ──> frontend container app (external ingress, nginx :8080)
//                    │  reverse-proxies /api, /predict, /evaluate, ...
//                    ▼
//                backend container app (INTERNAL ingress only, uvicorn :8000)
//                    │
//                    ▼
//                Supabase (hosted, outside Azure)
//
// The ML API has no public endpoint. That is the main security decision encoded
// here: the only way in is through the frontend's proxy, so the model, the
// registry admin route and the history endpoints are not directly addressable
// from the internet. The Flutter mobile shell therefore points at the frontend's
// FQDN too, and gets proxied like any browser.
//
// Deploy:
//   az deployment group create -g <rg> -f infra/main.bicep -p @infra/main.parameters.json
// ─────────────────────────────────────────────────────────────────────────────

targetScope = 'resourceGroup'

@description('Short environment name; becomes part of every resource name.')
@allowed(['staging', 'production'])
param environmentName string

@description('Azure region. Defaults to the resource group\'s location.')
param location string = resourceGroup().location

@description('Base name for resources. Keep short — ACR names allow 5-50 alphanumerics only.')
@minLength(3)
@maxLength(11)
param appName string = 'priceref'

@description('Container image tag to deploy. Always a git SHA, never "latest" — see note below.')
param imageTag string

@description('Supabase project URL, e.g. https://abc.supabase.co')
param supabaseUrl string = ''

@description('Supabase publishable anon key. Public by design; protected by RLS.')
@secure()
param supabaseAnonKey string = ''

@description('Supabase legacy HS256 JWT secret. Leave empty for projects on asymmetric keys.')
@secure()
param supabaseJwtSecret string = ''

@description('Backend replica bounds. minReplicas=1 keeps a model loaded and avoids cold starts.')
@minValue(0)
@maxValue(10)
param backendMinReplicas int = 1

@minValue(1)
@maxValue(30)
param backendMaxReplicas int = 3

var isProduction = environmentName == 'production'
var resourceSuffix = '${appName}-${environmentName}'
// ACR names disallow hyphens.
var registryName = toLower(replace('${appName}acr${environmentName}', '-', ''))

var backendAppName = '${resourceSuffix}-backend'
var frontendAppName = '${resourceSuffix}-frontend'

var commonTags = {
  application: 'priceref'
  environment: environmentName
  managedBy: 'bicep'
}

// ── Observability ────────────────────────────────────────────────────────────

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${resourceSuffix}'
  location: location
  tags: commonTags
  properties: {
    sku: { name: 'PerGB2018' }
    // 30 days in staging is plenty and keeps the bill down; production keeps a
    // quarter so a slow-burning model regression is still investigable.
    retentionInDays: isProduction ? 90 : 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${resourceSuffix}'
  location: location
  tags: commonTags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// ── Container registry ───────────────────────────────────────────────────────

resource registry 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: registryName
  location: location
  tags: commonTags
  sku: {
    // Basic is sufficient: two images, one region, no geo-replication. The
    // backend image is ~1.3 GB, comfortably inside Basic's 10 GB included storage.
    name: 'Basic'
  }
  properties: {
    // Admin user disabled — the container apps authenticate to ACR with their
    // managed identities, so no registry password exists to leak or rotate.
    adminUserEnabled: false
  }
}

// ── Identity ─────────────────────────────────────────────────────────────────
// One user-assigned identity shared by both apps, so the AcrPull grant is made
// once. A system-assigned identity would create a chicken-and-egg problem: the
// app cannot start without pull rights, and the grant cannot be made until the
// app (and therefore its identity) exists.

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${resourceSuffix}'
  location: location
  tags: commonTags
}

var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d' // AcrPull

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: registry
  // Deterministic GUID so redeploying is idempotent rather than erroring on a
  // duplicate assignment.
  name: guid(registry.id, identity.id, acrPullRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ── Container Apps environment ───────────────────────────────────────────────

resource containerAppEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${resourceSuffix}'
  location: location
  tags: commonTags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    zoneRedundant: false
  }
}

// ── Backend ──────────────────────────────────────────────────────────────────

resource backend 'Microsoft.App/containerApps@2024-03-01' = {
  name: backendAppName
  location: location
  tags: commonTags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  dependsOn: [acrPull]
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        // The critical line: internal only. The ML API is unreachable from the
        // internet and is addressable solely as
        // https://<name>.internal.<env-default-domain> from inside this environment.
        external: false
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
        traffic: [ { latestRevision: true, weight: 100 } ]
      }
      registries: [
        { server: registry.properties.loginServer, identity: identity.id }
      ]
      secrets: [
        { name: 'supabase-anon-key', value: supabaseAnonKey }
        { name: 'supabase-jwt-secret', value: supabaseJwtSecret }
      ]
      // Single revision mode: a new revision fully replaces the old one. The
      // alternative (multiple) would run two model-loading replicas
      // simultaneously and double the memory footprint during every deploy.
      activeRevisionsMode: 'Single'
    }
    template: {
      containers: [
        {
          name: 'backend'
          image: '${registry.properties.loginServer}/priceref-backend:${imageTag}'
          resources: {
            // ACA only accepts specific CPU/memory pairs; 1.0/2Gi is one of them.
            // The ensemble plus segment models occupy roughly 250-400 MB resident,
            // and pandas feature construction spikes above that, so 1 Gi is too
            // tight to be safe.
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            { name: 'PORT', value: '8000' }
            { name: 'APP_ENVIRONMENT', value: environmentName }
            { name: 'ACTIVE_VARIANT_ID', value: 'variant_1' }
            { name: 'LOG_LEVEL', value: isProduction ? 'info' : 'debug' }
            // Browser traffic arrives same-origin via the frontend proxy, so no
            // browser origin needs listing. This covers the Flutter shell, which
            // calls the frontend FQDN directly.
            { name: 'CORS_ALLOWED_ORIGINS', value: 'https://${frontendAppName}.${containerAppEnv.properties.defaultDomain}' }
            { name: 'SUPABASE_URL', value: supabaseUrl }
            { name: 'SUPABASE_ANON_KEY', secretRef: 'supabase-anon-key' }
            { name: 'SUPABASE_JWT_SECRET', secretRef: 'supabase-jwt-secret' }
            // Never enabled in a multi-replica deployment: each replica would
            // switch independently and serve a different model from its peers.
            { name: 'ALLOW_RUNTIME_VARIANT_SWITCH', value: 'false' }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
          ]
          probes: [
            {
              // Startup probe with a long budget: loading ~250 MB of model
              // artifacts off disk takes tens of seconds. Without this, the
              // liveness probe kills the container before it ever finishes
              // booting, producing an endless restart loop.
              type: 'Startup'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 15
              periodSeconds: 10
              failureThreshold: 30   // up to 5 minutes to become ready
              timeoutSeconds: 5
            }
            {
              type: 'Readiness'
              httpGet: { path: '/health', port: 8000 }
              periodSeconds: 15
              failureThreshold: 3
              timeoutSeconds: 5
            }
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              periodSeconds: 30
              failureThreshold: 3
              timeoutSeconds: 10
            }
          ]
        }
      ]
      scale: {
        // minReplicas: 1 rather than 0. Scale-to-zero would be cheaper, but the
        // first request after an idle period would wait through a full model
        // load — tens of seconds — which reads as a broken app.
        minReplicas: backendMinReplicas
        maxReplicas: backendMaxReplicas
        rules: [
          {
            name: 'http-concurrency'
            http: {
              // Deliberately low. Each replica runs one uvicorn worker with
              // single-threaded BLAS, so it handles a handful of concurrent
              // predictions before latency climbs. Scale out early.
              metadata: { concurrentRequests: '20' }
            }
          }
        ]
      }
    }
  }
}

// ── Frontend ─────────────────────────────────────────────────────────────────

resource frontend 'Microsoft.App/containerApps@2024-03-01' = {
  name: frontendAppName
  location: location
  tags: commonTags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  dependsOn: [acrPull]
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8080
        transport: 'auto'
        allowInsecure: false
        traffic: [ { latestRevision: true, weight: 100 } ]
      }
      registries: [
        { server: registry.properties.loginServer, identity: identity.id }
      ]
      secrets: [
        { name: 'supabase-anon-key', value: supabaseAnonKey }
      ]
      activeRevisionsMode: 'Single'
    }
    template: {
      containers: [
        {
          name: 'frontend'
          image: '${registry.properties.loginServer}/priceref-frontend:${imageTag}'
          resources: {
            // Serving static files and proxying; nginx needs very little.
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          env: [
            // Internal FQDN of the backend. https:// because ACA terminates TLS
            // at its internal ingress too.
            { name: 'BACKEND_ORIGIN', value: 'https://${backendAppName}.internal.${containerAppEnv.properties.defaultDomain}' }
            // Empty: the browser calls the API same-origin through the proxy.
            { name: 'APP_API_URL', value: '' }
            { name: 'SUPABASE_URL', value: supabaseUrl }
            { name: 'SUPABASE_ANON_KEY', secretRef: 'supabase-anon-key' }
            { name: 'APP_ENVIRONMENT', value: environmentName }
            { name: 'APP_RELEASE', value: imageTag }
          ]
          probes: [
            {
              type: 'Readiness'
              httpGet: { path: '/healthz', port: 8080 }
              initialDelaySeconds: 3
              periodSeconds: 10
              failureThreshold: 3
            }
            {
              type: 'Liveness'
              httpGet: { path: '/healthz', port: 8080 }
              periodSeconds: 30
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 5
        rules: [
          { name: 'http-concurrency', http: { metadata: { concurrentRequests: '100' } } }
        ]
      }
    }
  }
}

// ── Outputs ──────────────────────────────────────────────────────────────────
// Consumed by .github/workflows/cd.yml for the post-deploy smoke test.

@description('Public URL of the application.')
output frontendUrl string = 'https://${frontend.properties.configuration.ingress.fqdn}'

@description('Internal-only backend FQDN. Not reachable from the internet.')
output backendInternalFqdn string = backend.properties.configuration.ingress.fqdn

output registryLoginServer string = registry.properties.loginServer
output backendAppName string = backend.name
output frontendAppName string = frontend.name
