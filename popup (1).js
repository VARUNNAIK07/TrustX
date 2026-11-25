const API = "http://127.0.0.1:5000";

const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const textEl = document.getElementById("txt");
const buttonEl = document.getElementById("analyse-btn");

checkHealth();
buttonEl.addEventListener("click", analyseText);

async function checkHealth() {
  try {
    const response = await fetch(`${API}/health`);
    if (!response.ok) {
      throw new Error("Backend unavailable");
    }

    statusEl.textContent = "Backend connected. Popup and page scanning are ready.";
    statusEl.className = "status on";
  } catch (error) {
    statusEl.textContent = "Backend offline. Start app.py before analysing text.";
    statusEl.className = "status off";
  }
}

async function analyseText() {
  const text = textEl.value.trim();
  if (!text) {
    renderError("Enter some text first.");
    return;
  }

  buttonEl.disabled = true;
  buttonEl.textContent = "Analysing...";

  try {
    const response = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    const isFake = data.prediction === "FAKE";
    resultEl.hidden = false;
    resultEl.className = isFake ? "result-fake" : "result-real";
    resultEl.innerHTML = `
      <strong>${data.prediction}</strong><br>
      Confidence: ${data.confidence}%<br>
      FastText score: ${Math.round((data.fasttext_probability || 0) * 100)}%<br>
      ${data.reasons.join(", ")}
    `;
  } catch (error) {
    renderError(error.message || "Prediction failed");
  } finally {
    buttonEl.disabled = false;
    buttonEl.textContent = "Analyse Text";
  }
}

function renderError(message) {
  resultEl.hidden = false;
  resultEl.className = "result-error";
  resultEl.textContent = message;
}
