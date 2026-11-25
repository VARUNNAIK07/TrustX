const API = "http://127.0.0.1:5000";
const analysedArticles = new WeakSet();

boot();

function boot() {
  scanTweets();

  const observer = new MutationObserver(() => {
    scanTweets();
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });
}

function scanTweets() {
  const articles = document.querySelectorAll('article[data-testid="tweet"]');
  articles.forEach((article) => {
    if (analysedArticles.has(article)) {
      return;
    }

    const textNode = article.querySelector('[data-testid="tweetText"]');
    if (!textNode) {
      return;
    }

    const text = textNode.innerText.trim();
    if (!text) {
      return;
    }

    analysedArticles.add(article);
    injectPendingBadge(article);
    analyseTweet(article, text).catch(() => {
      renderBadge(article, {
        prediction: "ERROR",
        confidence: 0,
        reasons: ["Backend unavailable"],
      });
    });
  });
}

async function analyseTweet(article, text) {
  const payload = {
    text,
    replies: readMetric(article, '[data-testid="reply"]'),
    retweets: readMetric(article, '[data-testid="retweet"]'),
    likes: readMetric(article, '[data-testid="like"]'),
  };

  const response = await fetch(`${API}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Prediction failed");
  }

  renderBadge(article, data);
}

function readMetric(article, selector) {
  const metricNode = article.querySelector(selector);
  if (!metricNode) {
    return 0;
  }

  const text = metricNode.innerText || metricNode.getAttribute("aria-label") || "0";
  const match = text.match(/[\d,.]+[kKmM]?/);
  return match ? match[0] : 0;
}

function injectPendingBadge(article) {
  const anchor = getAnchor(article);
  if (!anchor || anchor.querySelector(".pc-badge")) {
    return;
  }

  const badge = document.createElement("div");
  badge.className = "pc-badge";
  badge.innerHTML = `
    <span class="pc-label pc-pending">Checking...</span>
    <span class="pc-reasons">Running hybrid model</span>
  `;
  anchor.appendChild(badge);
}

function renderBadge(article, data) {
  const anchor = getAnchor(article);
  if (!anchor) {
    return;
  }

  const existing = anchor.querySelector(".pc-badge");
  const badge = existing || document.createElement("div");
  const isFake = data.prediction === "FAKE";
  const isError = data.prediction === "ERROR";

  badge.className = "pc-badge";
  badge.innerHTML = `
    <span class="pc-label ${isError ? "pc-error" : isFake ? "pc-fake" : "pc-real"}">
      ${data.prediction}
    </span>
    <span class="pc-score">${Math.round(data.confidence || 0)}%</span>
    <span class="pc-reasons">${(data.reasons || []).join(", ")}</span>
  `;

  if (!existing) {
    anchor.appendChild(badge);
  }
}

function getAnchor(article) {
  return article.querySelector('[role="group"]')?.parentElement || article;
}
