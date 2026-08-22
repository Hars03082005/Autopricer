import { useState, useRef, useEffect } from 'react';
import { useApp } from '../context/AppContext.jsx';
import Icon from '../components/Icon.jsx';

const fmt = (n) => {
  const v = Number(n || 0);
  if (v >= 10000000) return `₹${(v / 10000000).toFixed(2)}Cr`;
  if (v >= 100000)   return `₹${(v / 100000).toFixed(2)}L`;
  if (v >= 1000)     return `₹${(v / 1000).toFixed(1)}k`;
  return `₹${Math.round(v).toLocaleString('en-IN')}`;
};

const QUICK_ACTIONS = [
  { label: 'Is this asking price fair?', query: 'Is this asking price fair for the condition and mileage?' },
  { label: 'What should I offer?', query: 'What should I offer as opening, target, and walk-away numbers?' },
  { label: 'What risks should I inspect?', query: 'What are the main risks and components I should inspect before buying?' },
  { label: 'How much profit can I make?', query: 'How much profit and ROI can I realistically expect on this deal?' },
];

function generateResponse(question, result, inputs) {
  const q = question.toLowerCase();
  const carName = `${inputs?.year || '2021'} ${inputs?.brand || 'Honda'} ${inputs?.model || 'City'}`;
  const price = result?.predictedPrice || 950000;
  const buyPrice = result?.recommendedBuyPrice || 840000;
  const profit = result?.expectedProfit || 78000;
  const margin = result?.expectedMarginPct || 10.5;
  const action = result?.action || 'BUY';
  const score = result?.dealQualityScore || 78;

  if (q.includes('fair') || q.includes('price') || q.includes('market value')) {
    return `For the **${carName}**, the ML baseline valuation is **${fmt(price)}** (Range: ${fmt(result?.priceMin || price * 0.94)} – ${fmt(result?.priceMax || price * 1.06)}).\n\nBased on comparable listings in ${inputs?.city || 'Bangalore'}, this estimate reflects average mileage of ${(Number(inputs?.mileage || 28000) / 1000).toFixed(0)}k km and ${inputs?.ownerCount || 1} prior owner(s). Any acquisition below **${fmt(buyPrice)}** delivers your target dealer margin.`;
  }

  if (q.includes('offer') || q.includes('negotiat') || q.includes('what should i')) {
    const floor = result?.opening_offer || Math.round((buyPrice * 0.95) / 500) * 500;
    const ceil = result?.max_offer || Math.round((buyPrice * 1.03) / 500) * 500;
    return `**Recommended Negotiation Protocol:**\n\n1. **Opening Anchor Offer:** ${fmt(floor)} — Anchor low citing reconditioning allowances and documentation fees.\n2. **Target Settlement:** **${fmt(buyPrice)}** — Secures your projected **+${fmt(profit)}** (${margin}% net ROI).\n3. **Walk-Away Threshold:** ${fmt(ceil)} — Do not exceed this ceiling to prevent margin erosion.`;
  }

  if (q.includes('risk') || q.includes('inspect') || q.includes('watch')) {
    const risks = result?.negativeFactors?.length ? result.negativeFactors : [
      'Inspect front suspension bushings and tyre tread depth (>40% life remaining)',
      'Check clutch pedal bite point and brake pad wear',
      'Verify service history records for timely fluid replacements',
    ];
    return `**Pre-Purchase Inspection Focus Areas (Risk Score: ${result?.riskScore || 32}/100):**\n\n${risks.map(r => `• ${r}`).join('\n')}\n\n*Recommendation:* Physical inspection required prior to releasing advance deposit.`;
  }

  if (q.includes('profit') || q.includes('margin') || q.includes('make') || q.includes('roi')) {
    return `**Financial Breakdown:**\n\n• **Acquisition Cost:** ${fmt(buyPrice)}\n• **Resale Benchmark:** ${fmt(result?.recommendedSellPrice || Math.round(price * 1.05))}\n• **Estimated Deductions (Recon, Holding, RTO):** −${fmt((result?.recon_cost || 18000) + (result?.holding_cost || 5000) + (result?.doc_cost || 4500))}\n• **Net Dealer Profit:** **+${fmt(profit)}** (${margin}% ROI)\n\n*Deal Call:* **${action}** (Quality Score: ${score}/100).`;
  }

  return `The **${carName}** currently holds a **${action}** decision indicator. Estimated market value is **${fmt(price)}** with target buy at **${fmt(buyPrice)}** to achieve **+${fmt(profit)}** profit. You can ask for negotiation ranges, risk inspections, or margin breakdowns.`;
}

function FormattedMessage({ text }) {
  if (!text) return null;
  const lines = text.split('\n');
  return (
    <div style={{ lineHeight: 1.55 }}>
      {lines.map((line, lIdx) => {
        const parts = line.split(/\*\*/g);
        return (
          <div key={lIdx} style={{ minHeight: line.trim() === '' ? 8 : undefined }}>
            {parts.map((part, pIdx) => {
              if (pIdx % 2 === 1) {
                return (
                  <strong key={pIdx} style={{ fontWeight: 800, color: 'var(--text-1)' }}>
                    {part}
                  </strong>
                );
              }
              return <span key={pIdx}>{part}</span>;
            })}
          </div>
        );
      })}
    </div>
  );
}

export default function AssistantScreen() {
  const { valuationResult, inputs, setActiveScreen } = useApp();
  const [messages, setMessages] = useState([
    {
      sender: 'ai',
      text: valuationResult
        ? `Ready to assist with **${inputs?.year} ${inputs?.brand} ${inputs?.model}**. Market value is estimated at **${fmt(valuationResult.predictedPrice)}** with a **${valuationResult.action || 'BUY'}** recommendation. Select a quick action below or ask a specific question.`
        : 'Welcome to the PriceRef Deal Assistant. Run a valuation on a vehicle to get instant AI negotiation strategies, risk inspections, and margin breakdowns.',
    },
  ]);
  const [inputVal, setInputVal] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = (text) => {
    const query = (text || inputVal).trim();
    if (!query) return;

    setMessages(prev => [
      ...prev,
      { sender: 'user', text: query },
      { sender: 'ai', text: generateResponse(query, valuationResult, inputs) },
    ]);
    setInputVal('');
  };

  return (
    <div className="screen" style={{ height: 'calc(100dvh - 52px)', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <div>
          <div className="page-title">AI Deal Assistant</div>
          <div className="page-subtitle">Contextual acquisition intelligence and negotiation strategy.</div>
        </div>
        {valuationResult && (
          <button className="btn btn-secondary btn-sm" onClick={() => setActiveScreen('result')}>
            <Icon name="bulb" size={13} strokeWidth={2} />
            <span>View Valuation Report</span>
          </button>
        )}
      </div>

      <div className="assistant-root">
        {/* Chat Stream Column */}
        <div className="assistant-chat-col">
          <div className="assistant-messages">
            {messages.map((m, idx) => (
              <div key={idx} className={`assistant-msg ${m.sender === 'user' ? 'user' : ''}`}>
                <div className={`msg-avatar ${m.sender === 'user' ? 'usr' : 'ai'}`}>
                  {m.sender === 'user' ? 'ME' : 'PR'}
                </div>
                <div className={`msg-bubble ${m.sender === 'user' ? 'usr' : 'ai'}`}>
                  <FormattedMessage text={m.text} />
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Prompts Bar */}
          <div style={{ padding: '8px 12px', background: 'var(--surface-2)', borderTop: '1px solid var(--border-2)', display: 'flex', gap: 6, overflowX: 'auto', scrollbarWidth: 'none' }}>
            {QUICK_ACTIONS.map((a, i) => (
              <button
                key={i}
                className="btn btn-secondary btn-sm"
                style={{ fontSize: 11.5, padding: '4px 10px', whiteSpace: 'nowrap' }}
                onClick={() => handleSend(a.query)}
              >
                {a.label}
              </button>
            ))}
          </div>

          {/* Input Box */}
          <form
            className="assistant-input-row"
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
          >
            <input
              type="text"
              className="assistant-input"
              placeholder="Ask about negotiation angles, risk checks, or profit scenarios..."
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
            />
            <button type="submit" className="btn btn-primary btn-sm">
              <Icon name="arrowRight" size={14} color="white" strokeWidth={2} />
              <span>Send</span>
            </button>
          </form>
        </div>

        {/* Right Column: Vehicle Context Card */}
        <div className="assistant-context-col">
          <div className="card">
            <div className="card-header">
              <div className="card-title">Vehicle Context</div>
              {valuationResult && (
                <span className={`badge ${valuationResult.action === 'BUY' ? 'badge-buy' : 'badge-caution'}`}>
                  {valuationResult.action || 'BUY'}
                </span>
              )}
            </div>

            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {valuationResult ? (
                <>
                  <div>
                    <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--text-1)' }}>
                      {inputs.year} {inputs.brand} {inputs.model}
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--text-4)', marginTop: 2 }}>
                      {inputs.fuel} · {inputs.transmission} · {Number(inputs.mileage || 0).toLocaleString('en-IN')} km
                    </div>
                  </div>

                  <div style={{ borderTop: '1px solid var(--border-2)', paddingTop: 10 }}>
                    <div style={{ fontSize: 10.5, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-4)' }}>
                      Market Value
                    </div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-1)' }}>
                      {fmt(valuationResult.predictedPrice)}
                    </div>
                  </div>

                  <div style={{ borderTop: '1px solid var(--border-2)', paddingTop: 10 }}>
                    <div style={{ fontSize: 10.5, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-4)' }}>
                      Target Buy Price
                    </div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: '#16a34a' }}>
                      {fmt(valuationResult.recommendedBuyPrice)}
                    </div>
                  </div>

                  <div style={{ borderTop: '1px solid var(--border-2)', paddingTop: 10 }}>
                    <div style={{ fontSize: 10.5, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-4)' }}>
                      Expected Profit
                    </div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: '#e85d26' }}>
                      +{fmt(valuationResult.expectedProfit)}
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ fontSize: 12.5, color: 'var(--text-4)', textAlign: 'center', padding: '20px 0' }}>
                  No vehicle active. Run a valuation to load acquisition context.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
