async function sendMessage(event) {
    event.preventDefault();
    const input = document.getElementById("userInput");
    const messageText = input.value.trim();
    if (!messageText) return;

    const messagesList = document.getElementById("messagesList");

    // Render User Message
    const userDiv = document.createElement("div");
    userDiv.className = "message user-msg";
    userDiv.innerHTML = `
        <div class="avatar">👤</div>
        <div class="msg-content">
            <p>${escapeHtml(messageText)}</p>
        </div>
    `;
    messagesList.appendChild(userDiv);
    input.value = "";
    messagesList.scrollTop = messagesList.scrollHeight;

    // Render Loading Indicator
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "message bot-msg";
    loadingDiv.innerHTML = `
        <div class="avatar">🧠</div>
        <div class="msg-content">
            <p><i>CMP-1.05B executing 24-layer forward pass over 1,059,878,400 parameters...</i></p>
        </div>
    `;
    messagesList.appendChild(loadingDiv);
    messagesList.scrollTop = messagesList.scrollHeight;

    try {
        const response = await fetch("http://localhost:9000/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: messageText })
        });

        const data = await response.json();
        messagesList.removeChild(loadingDiv);

        // Render Bot Response
        const botDiv = document.createElement("div");
        botDiv.className = "message bot-msg";
        
        let formattedResponse = escapeHtml(data.response);
        if (data.response.includes("\n") || data.response.includes("def ")) {
            formattedResponse = `<pre>${escapeHtml(data.response)}</pre>`;
        } else {
            formattedResponse = `<p>${formattedResponse}</p>`;
        }

        botDiv.innerHTML = `
            <div class="avatar">🧠</div>
            <div class="msg-content">
                <strong>${data.model}</strong>
                ${formattedResponse}
            </div>
        `;
        messagesList.appendChild(botDiv);

        // Update Memory State Bar
        if (data.state_norm !== undefined) {
            const normVal = document.getElementById("normVal");
            const normFill = document.getElementById("normFill");
            normVal.textContent = `Ephemeral Norm: ${data.state_norm.toFixed(4)}`;
            const fillPct = Math.min(Math.max((data.state_norm / 2.0) * 100, 10), 100);
            normFill.style.width = `${fillPct}%`;
        }

        messagesList.scrollTop = messagesList.scrollHeight;
    } catch (err) {
        messagesList.removeChild(loadingDiv);
        const errDiv = document.createElement("div");
        errDiv.className = "message bot-msg";
        errDiv.innerHTML = `
            <div class="avatar">⚠️</div>
            <div class="msg-content" style="color: #f87171;">
                <p>Error connecting to CMP-1B backend on http://localhost:9000. Check server logs.</p>
            </div>
        `;
        messagesList.appendChild(errDiv);
        messagesList.scrollTop = messagesList.scrollHeight;
    }
}

function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
