import { useState, useRef, useEffect } from 'react';
import { useApp } from '../context/AppContext.jsx';
import Icon from '../components/Icon.jsx';

const fmtL = (n) => {
  const v = Number(n || 0);
  if (v >= 100000) return `₹${(v/100000).toFixed(2)}L`;
  return `₹${Math.round(v).toLocaleString()}`;
};

const SUGGESTIONS = [
  'What is the ML market value for this vehicle?',
  'Should I buy this car at the asking price?',
  'What are the biggest risk factors?',
  'Explain the depreciation for this car',
  'What is the realistic dealer profit?',
  'Compare this to similar vehicles',
];

function buildContext(valuationResult, inputs) {
  if (!valuationResult) return 'No vehicle has been evaluated yet. Ask me to explain PricerPoint features.';
  return `
Vehicle: ${inputs?.year} ${inputs?.brand} ${inputs?.model} (${inputs?.variant || 'base'})
Fuel: ${inputs?.fuel} · Transmission: ${inputs?.transmission}
Odometer: ${Number(inputs?.mileage||0).toLocaleString()} km
City: ${inputs?.city} · Owners: ${inputs?.ownerCount}
Condition: ${inputs?.condition}

ML Results:
Market Value: ${fmtL(valuationResult.predictedPrice)}
Price Range: ${fmtL(valuationResult.priceMin)} – ${fmtL(valuationResult.priceMax)}
Confidence: ${valuationResult.confidenceScore}%
Segment: ${valuationResult.segmentClass?.toUpperCase()}
Recommendation: ${valuationResult.action}
Buy Price: ${fmtL(valuationResult.recommendedBuyPrice)}
Sell Price: ${fmtL(valuationResult.recommendedSellPrice)}
Expected Profit: ${fmtL(valuationResult.expectedProfit)}
Margin: ${valuationResult.expectedMarginPct}%
Risk Score: ${valuationResult.riskScore}/100 (${valuationResult.riskLevel})

Positive factors: ${(valuationResult.positiveFactors||[]).join(', ')}
Risk factors: ${(valuationResult.negativeFactors||[]).join(', ')}
Warnings: ${(valuationResult.warnings||[]).join(', ')}
  `.trim();
}

function generateResponse(question, context, result, inputs) {
  const q = question.toLowerCase();

  if (q.includes('market value') || q.includes('price') || q.includes('worth')) {
    return `Based on the ML model, this **${inputs?.year} ${inputs?.brand} ${inputs?.model}** has a market value of **${fmtL(result?.predictedPrice)}** with ${result?.confidenceScore}% confidence. The price range is ${fmtL(result?.priceMin)} to ${fmtL(result?.priceMax)}.`;
  }
  if (q.includes('buy') || q.includes('should i') || q.includes('recommend')) {
    const action = result?.action || 'MANUAL REVIEW';
    const emoji = action === 'BUY' ? '✅' : action === 'NEGOTIATE' ? '🔶' : '❌';
    return `${emoji} The ML engine recommends: **${action}**\n\nIdeal buy price: **${fmtL(result?.recommendedBuyPrice)}**. Expected dealer profit: **${fmtL(result?.expectedProfit)}** (${result?.expectedMarginPct}% margin).\n\n${action === 'BUY' ? 'This is a good opportunity — move quickly.' : action === 'NEGOTIATE' ? 'You can potentially close this deal if you negotiate the price down.' : 'This deal has too much risk or insufficient margin for a profitable flip.'}`;
  }
  if (q.includes('risk') || q.includes('danger') || q.includes('concern')) {
    const risks = result?.negativeFactors || [];
    const score = result?.riskScore || 0;
    return `Risk score: **${score}/100** (${result?.riskLevel || 'Medium'})\n\n**Key risk factors:**\n${risks.map(r => `• ${r}`).join('\n') || '• No major risks identified'}\n\n${score > 65 ? '⚠️ High risk — ensure a thorough inspection before buying.' : score > 35 ? '🔶 Moderate risk — manageable with proper due diligence.' : '✅ Low risk — this vehicle has a clean profile.'}`;
  }
  if (q.includes('depreciation') || q.includes('age') || q.includes('year')) {
    const age = new Date().getFullYear() - Number(inputs?.year || 2020);
    return `This **${age}-year-old** ${inputs?.brand} ${inputs?.model} has depreciated from its original price. At **${(Number(inputs?.mileage||0)/1000).toFixed(0)}k km**, the vehicle has experienced ${age * 8}–${age * 12}% depreciation from showroom price.\n\nML Market Value: **${fmtL(result?.predictedPrice)}**`;
  }
  if (q.includes('profit') || q.includes('margin') || q.includes('earn')) {
    return `**Dealer profit analysis:**\n\n• Buy price: ${fmtL(result?.recommendedBuyPrice)}\n• Sell price: ${fmtL(result?.recommendedSellPrice)}\n• Expected profit: **${fmtL(result?.expectedProfit)}**\n• Margin: **${result?.expectedMarginPct}%**\n\nFor mass-market cars, a healthy dealer profit is ₹25,000–₹80,000. Go to **Pricing** tab for a full cost breakdown.`;
  }
  if (q.includes('compar') || q.includes('similar') || q.includes('alternative')) {
    return `I can't fetch live market listings, but the **${result?.segmentClass?.toUpperCase()}** segment model was trained on 213,820 Indian used car transactions. The ML confidence of **${result?.confidenceScore}%** indicates how representative this vehicle is within its segment.\n\nCheck the **Pricing** tab for comparables from your own evaluation history.`;
  }
  if (q.includes('explain') || q.includes('how') || q.includes('why')) {
    return `The ML model used **${result?.segmentClass?.toUpperCase()}** segment-specific training to predict this price.\n\n**Top value drivers:**\n${(result?.positiveFactors||[]).slice(0,3).map(f=>`✓ ${f}`).join('\n') || '—'}\n\n**Risk factors:**\n${(result?.negativeFactors||[]).slice(0,3).map(f=>`✗ ${f}`).join('\n') || '—'}\n\nVisit **AI Explain** for a full SHAP-style feature impact analysis.`;
  }

  // Default helpful response
  return `I'm your PricerPoint dealer assistant. I can help you with:\n\n• **Interpreting** the ML valuation result\n• **Buy/sell decision** analysis\n• **Risk factor** breakdown\n• **Profit and margin** estimates\n• **Depreciation** context\n\nFor this vehicle (${inputs?.brand} ${inputs?.model}): **${fmtL(result?.predictedPrice)}** market value · **${result?.action}** recommendation.\n\nWhat would you like to know?`;
}

export default function AssistantScreen() {
  const { valuationResult, inputs } = useApp();
  const [messages, setMessages] = useState([
    {
      role: 'ai',
      text: valuationResult
        ? `Hello! I'm your PricerPoint assistant. I've analysed the **${inputs?.year} ${inputs?.brand} ${inputs?.model}** — market value **${fmtL(valuationResult?.predictedPrice)}**, recommendation: **${valuationResult?.action}**. What would you like to know?`
        : `Hello! I'm your PricerPoint dealer assistant. Run a vehicle valuation first, then I can answer detailed questions about the ML result, risk factors, and dealer margin. What can I help with?`,
    },
  ]);
  const [input, setInput]     = useState('');
  const [typing, setTyping]   = useState(false);
  const messagesEndRef         = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  const handleSend = async (text) => {
    const question = text || input.trim();
    if (!question) return;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: question }]);
    setTyping(true);

    await new Promise(r => setTimeout(r, 800 + Math.random() * 700));

    const context = buildContext(valuationResult, inputs);
    const response = generateResponse(question, context, valuationResult, inputs);

    setTyping(false);
    setMessages(prev => [...prev, { role: 'ai', text: response }]);
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  // Format message with bold (**text**)
  const formatMsg = (text) => {
    const parts = text.split(/\*\*(.*?)\*\*/g);
    return parts.map((part, i) =>
      i % 2 === 1
        ? <strong key={i}>{part}</strong>
        : part.split('\n').map((line, j) => (
            <span key={`${i}-${j}`}>{line}{j < part.split('\n').length-1 && <br />}</span>
          ))
    );
  };

  return (
    <div className="screen">
      <div className="page-header" style={{ marginBottom:16 }}>
        <div>
          <div className="page-title">AI Assistant</div>
          <div className="page-subtitle">Ask questions about the valuation, risk, or dealer strategy</div>
        </div>
        <span className="badge badge-info">Beta</span>
      </div>

      <div className="card" style={{ padding:0, overflow:'hidden' }}>
        {/* Messages */}
        <div
          className="chat-messages"
          style={{ padding:'16px 16px 0', maxHeight:'55vh', overflowY:'auto' }}
        >
          {messages.map((msg, i) => (
            <div key={i} className={`chat-bubble-wrap ${msg.role}`}>
              {msg.role === 'ai' && (
                <div className="chat-avatar ai">
                  <Icon name="brain" size={14} color="white" strokeWidth={2} />
                </div>
              )}
              <div className={`chat-bubble ${msg.role}`}>
                {formatMsg(msg.text)}
              </div>
            </div>
          ))}

          {typing && (
            <div className="chat-bubble-wrap">
              <div className="chat-avatar ai">
                <Icon name="brain" size={14} color="white" strokeWidth={2} />
              </div>
              <div className="chat-bubble ai" style={{ display:'flex', gap:6, alignItems:'center', padding:'14px 18px' }}>
                {[0,1,2].map(i => (
                  <div key={i} className="splash-dots" style={{ margin:0 }}>
                    <span style={{ animationDelay:`${i*0.2}s`, width:7, height:7, background:'#94a3b8' }} />
                  </div>
                ))}
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Suggestions */}
        {messages.length === 1 && (
          <div className="chat-suggestions" style={{ padding:'12px 16px 0' }}>
            {SUGGESTIONS.slice(0, 4).map((s, i) => (
              <button key={i} className="chat-suggestion-pill" onClick={() => handleSend(s)}>
                {s}
              </button>
            ))}
          </div>
        )}

        {/* Input row */}
        <div className="chat-input-row" style={{ padding:'12px 16px 16px' }}>
          <input
            className="chat-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about this vehicle…"
            disabled={typing}
          />
          <button
            className="chat-send-btn"
            onClick={() => handleSend()}
            disabled={!input.trim() || typing}
          >
            <Icon name="arrowRight" size={16} color="white" strokeWidth={2.2} />
          </button>
        </div>
      </div>
    </div>
  );
}
