import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist', 'mobile', 'mobile/**', 'venv', 'venv/**']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
  },

  // ── Pre-existing debt ─────────────────────────────────────────────────────
  // `npm run lint` reported 34 errors before this change, which would have made
  // it useless as a CI gate — the pipeline would fail on day one for reasons
  // unrelated to whatever was being merged.
  //
  // Rather than skip the gate, or sweep files this change does not otherwise
  // touch, each rule is downgraded to a warning for the exact files that were
  // already failing. New and edited code is held to the full standard, and
  // `--max-warnings` in CI stops the count from growing.
  //
  // Removing an entry here is a self-contained cleanup. Counts as of this commit:
  {
    files: ['src/screens/PricingScreen.jsx'],
    rules: { 'no-unused-vars': 'warn' },                    // 12x
  },
  {
    files: ['src/screens/InputScreen.jsx'],
    rules: {
      'no-useless-escape': 'warn',                          // 8x — regex escapes
      'no-unused-vars': 'warn',                             // 4x
      'react-hooks/set-state-in-effect': 'warn',            // 1x
    },
  },
  {
    files: ['src/screens/ResultScreen.jsx'],
    rules: {
      'no-unused-vars': 'warn',                             // 4x
      'react-hooks/preserve-manual-memoization': 'warn',    // 1x
    },
  },
  {
    files: [
      'src/screens/EnhancedResultScreen.jsx',
      'src/components/WheelrPanels.jsx',
      'src/utils/wheelrCosts.js',
    ],
    rules: { 'no-unused-vars': 'warn' },                    // 1x each
  },
  {
    files: ['src/components/SearchableDropdown.jsx'],
    rules: { 'react-hooks/set-state-in-effect': 'warn' },   // 1x
  },
])
