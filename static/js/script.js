/* =============================================
   FSSM Chatbot — script.js
   ============================================= */

(function () {
  'use strict';

  // ── DOM refs ──────────────────────────────────
  const messagesArea  = document.getElementById('messages-area');
  const userInput     = document.getElementById('user-input');
  const sendBtn       = document.getElementById('send-btn');
  const welcomeBlock  = document.getElementById('welcome-block');

  // ── State ─────────────────────────────────────
  let isLoading = false;

  // ── Init ──────────────────────────────────────
  userInput.addEventListener('input', onInput);
  userInput.addEventListener('keydown', onKeydown);
  sendBtn.addEventListener('click', sendMessage);

  // Auto-resize textarea
  function onInput() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    sendBtn.disabled = userInput.value.trim() === '' || isLoading;
  }

  function onKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && userInput.value.trim()) sendMessage();
    }
  }

  // ── Suggestion chips / welcome buttons ───────
  document.addEventListener('click', (e) => {
    const chip = e.target.closest('[data-query]');
    if (chip) {
      const q = chip.dataset.query;
      userInput.value = q;
      onInput();
      sendMessage();
    }
  });

  // ── Send ──────────────────────────────────────
  async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || isLoading) return;

    // Hide welcome block on first message
    if (welcomeBlock) {
      welcomeBlock.style.animation = 'fadeSlideUp .25s ease reverse both';
      setTimeout(() => welcomeBlock.remove(), 250);
    }

    // Append user bubble
    appendMessage('user', text);

    // Reset input
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;
    isLoading = true;

    // Show typing indicator
    const typingEl = appendTyping();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text }),
      });

      const data = await res.json();
      typingEl.remove();

      if (data.error) {
        appendMessage('bot', '⚠️ ' + (data.error || 'Une erreur est survenue.'));
      } else {
        appendMessage('bot', data.answer, data.sources || []);
      }

    } catch (err) {
      typingEl.remove();
      appendMessage('bot', '⚠️ Impossible de joindre le serveur. Veuillez réessayer.');
    } finally {
      isLoading = false;
      onInput();
      userInput.focus();
    }
  }

  // ── DOM helpers ───────────────────────────────

  function appendMessage(role, text, sources = []) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = role === 'bot' ? 'FS' : 'Moi';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = formatText(text);

    if (sources.length > 0) {
      const sourcesDiv = document.createElement('div');
      sourcesDiv.className = 'msg-sources';
      sources.slice(0, 4).forEach(src => {
        const tag = document.createElement('span');
        tag.className = 'source-tag';
        const label = src.source || src.filename || src.title || 'Source';
        tag.textContent = '📎 ' + truncate(label, 28);
        sourcesDiv.appendChild(tag);
      });
      bubble.appendChild(sourcesDiv);
    }

    msg.appendChild(avatar);
    msg.appendChild(bubble);
    messagesArea.appendChild(msg);
    scrollBottom();
    return msg;
  }

  function appendTyping() {
    const msg = document.createElement('div');
    msg.className = 'message bot';
    msg.innerHTML = `
      <div class="msg-avatar">FS</div>
      <div class="msg-bubble">
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>`;
    messagesArea.appendChild(msg);
    scrollBottom();
    return msg;
  }

  function scrollBottom() {
    messagesArea.scrollTo({ top: messagesArea.scrollHeight, behavior: 'smooth' });
  }

  // ── Text formatting (minimal markdown) ────────
  function formatText(text) {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  }

  function truncate(str, n) {
    return str.length > n ? str.slice(0, n - 1) + '…' : str;
  }

  // Initial send btn state
  sendBtn.disabled = true;

})();
