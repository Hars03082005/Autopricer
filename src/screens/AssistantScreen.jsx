import { useState, useRef, useEffect } from 'react';
import { useApp } from '../context/AppContext.jsx';
import Icon from '../components/Icon.jsx';
const fmtL = (n) => {
  const v = Number(n || 0);
  if (v >= 100000) return `₹${(v/100000).toFixed(2)}L`;
  return `₹${Math.round(v).toLocaleString()}`;
};
const SUGGESTIONS = [
  'Is this a good deal at the asking price?',
  'What should I offer to buy this car?',
  'What are the main risks I should watch out for?',
  'How much profit can I realistically make?',
  'Why is the price estimated at this level?',
  'How does mileage affect the value here?',
];
function buildContext(valuationResult, inputs) {
  if (!valuationResult) return 'No vehicle has been evaluated yet. Ask me to explain PriceRef features.';
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
    return `The **${inputs?.year} ${inputs?.brand} ${inputs?.model}** is valued at **${fmtL(result?.predictedPrice)}** — model confidence is ${result?.confidenceScore}%. Realistic range is ${fmtL(result?.priceMin)} to ${fmtL(result?.priceMax)} depending on negotiation and condition on the day.`;
  }
  if (q.includes('buy') || q.includes('should i') || q.includes('recommend') || q.includes('offer') || q.includes('deal')) {
    const action = result?.action || 'MANUAL REVIEW';
    const emoji = action === 'BUY' ? '✅' : action === 'NEGOTIATE' ? '🔶' : '❌';
    return `${emoji} **${action}**\n\nTarget buy price: **${fmtL(result?.recommendedBuyPrice)}**. If you can get in at that number, you're looking at **${fmtL(result?.expectedProfit)}** profit (${result?.expectedMarginPct}% margin) once the car sells.\n\n${action === 'BUY' ? 'Numbers work. Worth moving on this one.' : action === 'NEGOTIATE' ? "There's room here, but you'll need to push the seller down. Don't pay asking." : "Margin is too thin or the risk is too high. Better to walk away."}`;
  }
  if (q.includes('risk') || q.includes('danger') || q.includes('concern') || q.includes('watch')) {
    const risks = result?.negativeFactors || [];
    const score = result?.riskScore || 0;
    return `Risk score is **${score}/100** (${result?.riskLevel || 'Medium'}).\n\n**Things to watch out for:**\n${risks.map(r => `• ${r}`).join('\n') || '• Nothing flagged as a major concern'}\n\n${score > 65 ? '⚠️ High risk — get a proper inspection done before committing.' : score > 35 ? '🔶 Moderate risk — manageable, but do your homework.' : '✅ Looks clean. Low-risk pick up if the price is right.'}`;
  }
  if (q.includes('depreciation') || q.includes('age') || q.includes('year')) {
    const age = new Date().getFullYear() - Number(inputs?.year || 2020);
    return `This **${age}-year-old** ${inputs?.brand} ${inputs?.model} has depreciated from its original price. At **${(Number(inputs?.mileage||0)/1000).toFixed(0)}k km**, the vehicle has experienced ${age * 8}–${age * 12}% depreciation from showroom price.\n\nMarket Value: **${fmtL(result?.predictedPrice)}**`;
  }
  if (q.includes('profit') || q.includes('margin') || q.includes('earn') || q.includes('make')) {
    return `Here's the rough math:\n\n• Buy at: ${fmtL(result?.recommendedBuyPrice)}\n• Sell at: ${fmtL(result?.recommendedSellPrice)}\n• Walk away with: **${fmtL(result?.expectedProfit)}** (${result?.expectedMarginPct}% margin)\n\nFor most mass-market cars, anything between ₹25K–₹80K is a solid flip. Hit the **Pricing** tab for a full breakdown of recon, holding, and transfer costs.`;
  }
  if (q.includes('compar') || q.includes('similar') || q.includes('alternative')) {
    return `I can't fetch live market listings, but the **${result?.segmentClass?.toUpperCase()}** segment model was trained on 213,820 Indian used car transactions. The confidence of **${result?.confidenceScore}%** shows how well this vehicle fits its segment.\n\nCheck the **Pricing** tab for comparables from your own evaluation history.`;
  }
  if (q.includes('explain') || q.includes('how') || q.includes('why') || q.includes('reason') || q.includes('mileage') || q.includes('age')) {
    return `The price was calculated using the **${result?.segmentClass?.toUpperCase()}** segment model, which was trained on cars in a similar value range.\n\n**What's pushing the price up:**\n${(result?.positiveFactors||[]).slice(0,3).map(f=>`✓ ${f}`).join('\n') || '—'}\n\n**What's dragging it down:**\n${(result?.negativeFactors||[]).slice(0,3).map(f=>`✗ ${f}`).join('\n') || '—'}\n\nCheck the **Explain** tab for a full feature-by-feature breakdown.`;
  }
  // Default helpful response
  return `Not sure what you're looking for — try asking about the offer price, risks, profit potential, or why the value came in where it did.\n\n${result ? `Quick summary for this **${inputs?.brand} ${inputs?.model}**: valued at **${fmtL(result?.predictedPrice)}**, call is **${result?.action}**.` : 'Run a valuation first and I can give you the full picture.'}`;
}
export default function AssistantScreen() {
  const { valuationResult, inputs } = useApp();
  const [messages, setMessages] = useState([
    {
      role: 'ai',
      text: valuationResult
        ? `I've looked at the **${inputs?.year} ${inputs?.brand} ${inputs?.model}**. Market value comes to **${fmtL(valuationResult?.predictedPrice)}** and the call is **${valuationResult?.action}**. What do you want to dig into?`
        : `Run a valuation first and I can walk you through the numbers — what to offer, what the risks are, and whether the deal makes sense. What's on your mind?`,
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
          <div className="page-title">Deal Assistant</div>
          <div className="page-subtitle">Ask anything about this vehicle — offer price, risks, or profit</div>
        </div>
        <span className="badge badge-info">Beta</span>
      </div>
      <div className="card" style={{ padding:0, overflow:'hidden' }}>
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
        {messages.length === 1 && (
          <div className="chat-suggestions" style={{ padding:'12px 16px 0' }}>
            {SUGGESTIONS.slice(0, 4).map((s, i) => (
              <button key={i} className="chat-suggestion-pill" onClick={() => handleSend(s)}>
                {s}
              </button>
            ))}
          </div>
        )}
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
