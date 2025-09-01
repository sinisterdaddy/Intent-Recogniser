// DOM elements
const chatBox = document.getElementById("chat-box");
const inputField = document.getElementById("input");
const sendButton = document.getElementById("send");
const modelSelector = document.getElementById("model");

// Append messages to chat box
function appendMessage(sender, message, isHTML = false) {
  const msg = document.createElement("div");
  msg.classList.add("message");
  msg.innerHTML = `<strong>${sender}:</strong> ${isHTML ? message : escapeHTML(message)}`;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// Escape user input (if not using innerHTML)
function escapeHTML(str) {
  return str.replace(/[&<>"']/g, (m) => {
    const escape = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    };
    return escape[m];
  });
}

// Handle send button click
sendButton.addEventListener("click", handleSend);

// Also allow pressing Enter
inputField.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleSend();
});

async function handleSend() {
  const userInput = inputField.value.trim();
  const model = modelSelector.value;

  if (!userInput) return;

  appendMessage("You", userInput);
  inputField.value = "";

  // Show temporary loading
  const loadingMsg = document.createElement("div");
  loadingMsg.classList.add("message");
  loadingMsg.id = "loading";
  loadingMsg.innerHTML = `<em>Bot is typing...</em>`;
  chatBox.appendChild(loadingMsg);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: userInput, model })
    });

    const data = await res.json();
    document.getElementById("loading").remove();

    const botResponse = `
    <div><strong>Intent:</strong> ${data.intent} (${(data.confidence * 100).toFixed(2)}%)</div>
    <div><strong>Response:</strong> ${data.response}</div>
`;
 
    appendMessage("Bot", botResponse, true);
  } catch (err) {
    console.error(err);
    document.getElementById("loading").remove();
    appendMessage("Bot", "⚠️ Something went wrong.");
  }
}
