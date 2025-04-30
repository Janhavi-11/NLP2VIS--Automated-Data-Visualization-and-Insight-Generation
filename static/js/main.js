// 🌙 Toggle dark/light mode
const themeToggleBtn = document.getElementById('themeToggler');
themeToggleBtn.addEventListener('click', function () {
  document.body.classList.toggle('dark-mode');
  const icon = this.querySelector('i');
  if (document.body.classList.contains('dark-mode')) {
    icon.classList.remove('bx-moon');
    icon.classList.add('bx-sun');
  } else {
    icon.classList.remove('bx-sun');
    icon.classList.add('bx-moon');
  }
});

// 🔽 Scroll helper
function scrollToBottom(container) {
  container.scrollTop = container.scrollHeight;
}

// ⏳ Show loading spinner/message
function showLoading(container) {
  const loader = document.createElement('div');
  loader.className = 'loading-message';
  loader.innerText = 'Generating chart...';
  container.appendChild(loader);
  scrollToBottom(container);
  return loader;
}

// ✅ Remove loading
function removeLoading(loader) {
  if (loader) loader.remove();
}

// ✨ Handle form submit
document.querySelector(".prompt__form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const inputField = document.querySelector(".prompt__form-input");
  const input = inputField.value.trim();
  const chatBox = document.getElementById("chat-box");
  const layout = document.getElementById("layout");
  const queryResult = document.getElementById("query-result");

  if (input !== "") {
    inputField.value = "";
    document.getElementById("welcome-wrapper").style.display = "none";
    layout.style.display = "flex";
    layout.classList.add("fullscreen-layout");

    const messageBlock = document.createElement("div");
    messageBlock.classList.add("message-block");

    const userMessage = document.createElement("div");
    userMessage.classList.add("user-message");
    userMessage.innerText = input;
    messageBlock.appendChild(userMessage);
    chatBox.appendChild(messageBlock);
    scrollToBottom(chatBox);

    const loader = showLoading(messageBlock);

    try {
      const response = await fetch("/generate_chart", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input })
      });

      const result = await response.json();
      removeLoading(loader);

      if (result.error) {
        const errorMessage = document.createElement("div");
        errorMessage.classList.add("error-message");
        errorMessage.innerText = result.error;
        messageBlock.appendChild(errorMessage);
        return;
      }

      // SQL block
      const sqlLabel = document.createElement("div");
      sqlLabel.classList.add("sql-query");
      messageBlock.appendChild(sqlLabel);

      // Chart block
      const chartWrapper = document.createElement("div");
      chartWrapper.classList.add("chart-wrapper");

      const chatChart = document.createElement("img");
      chatChart.src = result.chart_url;
      chatChart.classList.add("chart-img");

      chartWrapper.appendChild(chatChart);
      messageBlock.appendChild(chartWrapper);

      // Sidebar preview
      const sidebarChart = document.createElement("img");
      sidebarChart.src = result.chart_url;
      sidebarChart.classList.add("chart-img");

      const resultWrapper = document.createElement("div");
      resultWrapper.classList.add("chart-wrapper");
      resultWrapper.appendChild(sidebarChart);
      queryResult.appendChild(resultWrapper);

      scrollToBottom(chatBox);

      // ✅ Clear the input field ONLY after successful chart render
      inputField.value = "";

    } catch (err) {
      console.error("Fetch error:", err);
      removeLoading(loader);
      const errorMessage = document.createElement("div");
      errorMessage.classList.add("error-message");
      errorMessage.innerText = "Something went wrong. Please try again.";
      messageBlock.appendChild(errorMessage);
    }
  }
});

// 🧽 Clear input field
document.getElementById("deleteButton").addEventListener("click", () => {
  document.querySelector(".prompt__form-input").value = "";
});

// 🔘 Suggested queries auto-fill
document.querySelectorAll('.suggests__item').forEach(item => {
  item.addEventListener('click', () => {
    const queryText = item.querySelector('p').textContent;
    document.querySelector('.prompt__form-input').value = queryText;
  });
});

// 💡 Enable Ctrl+Enter or Cmd+Enter to submit
document.querySelector('.prompt__form-input').addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    document.querySelector('.prompt__form').dispatchEvent(new Event('submit'));
  }
});

// 💬 Emoji suggestion toggle as true switch
const emojiBtn = document.getElementById("emojiToggleBtn");
const emojiSuggestions = document.getElementById("emojiSuggestions");

emojiBtn?.addEventListener('click', () => {
  emojiBtn.classList.toggle('active');
  emojiSuggestions.style.display = emojiBtn.classList.contains('active') ? 'flex' : 'none';
});

// 🧭 Guided tour trigger
document.getElementById("info-icon").addEventListener("click", () => {
  const queryResult = document.getElementById("query-result");
  const wasHidden = window.getComputedStyle(queryResult).display === "none";

  if (wasHidden) {
    queryResult.style.display = "block";
    queryResult.style.minHeight = "50px";
  }

  setTimeout(() => {
    const tour = introJs().setOptions({
      steps: [
        {
          intro: `
            <div class="tour-welcome">
              <h2>👋 Welcome to <span class="highlight">NLP2VIS</span>!</h2>
              <p>Let me walk you through how to generate cool charts with simple language.</p>
            </div>
          `
        },
        {
          element: document.querySelector(".suggests__item"),
          intro: "Try clicking on a suggestion to auto-fill the input box.",
        },
        {
          element: document.querySelector(".prompt__form-input"),
          intro: "Type your own natural language query here.",
        },
        {
          element: document.querySelector(".prompt__form-button"),
          intro: "Click this send button to generate a chart.",
        },
        {
          element: document.getElementById("deleteButton"),
          intro: "Click this to clear the query entered.",
        },
        {
          element: queryResult,
          intro: "Your generated chart and SQL query will appear here.",
          scrollTo: 'element'
        }
      ],
      scrollToElement: false,
      scrollTo: 'off',
      nextLabel: `<i class='bx bx-right-arrow-circle'></i>`,
      prevLabel: `<i class='bx bx-left-arrow-circle'></i>`,
      doneLabel: "Finish"
    });

    tour.oncomplete(() => {
      if (wasHidden) {
        queryResult.style.display = "none";
        queryResult.style.minHeight = "";
      }
    });

    tour.onexit(() => {
      if (wasHidden) {
        queryResult.style.display = "none";
        queryResult.style.minHeight = "";
      }
    });

    tour.start();
  }, 200);
});

// Zoom Button
let zoomLevel = 1;

function openChartModal(chartElement, messageElement) {
  const modal = document.getElementById('chartModal');
  const modalMessage = document.getElementById('modalMessage');
  const modalChart = document.getElementById('modalChartContainer');

  modalMessage.innerHTML = messageElement.innerHTML;
  modalChart.innerHTML = chartElement.innerHTML;

  const img = modalChart.querySelector('img');
  if (img) {
    img.id = 'modalChartImage';
    img.style.transform = 'scale(1)';
  }

  if (!document.querySelector('.close-btn')) {
    const closeBtn = document.createElement('button');
    closeBtn.classList.add('close-btn');
    closeBtn.innerHTML = '&times;';
    closeBtn.onclick = closeChartModal;
    document.querySelector('.modal-content').appendChild(closeBtn);
  }

  modal.classList.remove('hidden');
}

function closeChartModal() {
  document.getElementById('chartModal').classList.add('hidden');
  const closeBtn = document.querySelector('.close-btn');
  if (closeBtn) closeBtn.remove();
}

document.addEventListener('DOMContentLoaded', function () {
  document.getElementById('zoomIn').addEventListener('click', () => {
    const chartImg = document.querySelector('#modalChartImage');
    if (!chartImg) return;
    zoomLevel += 0.25;
    chartImg.style.transform = `scale(${zoomLevel})`;
  });

  document.getElementById('zoomOut').addEventListener('click', () => {
    const chartImg = document.querySelector('#modalChartImage');
    if (!chartImg) return;
    zoomLevel = Math.max(1, zoomLevel - 0.25);
    chartImg.style.transform = `scale(${zoomLevel})`;
  });
});

function addChartToChat(imgSrc, messageHTML = '') {
  const chatBox = document.getElementById('chat-box');

  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble user';

  const messageBlock = document.createElement('div');
  messageBlock.className = 'message-block';
  messageBlock.innerHTML = messageHTML;

  const chartWrapper = document.createElement('div');
  chartWrapper.className = 'chat-chart-wrapper chart-wrapper';
  chartWrapper.innerHTML = `<img src="${imgSrc}" alt="Chart" class="chart-img" />`;

  bubble.appendChild(messageBlock);
  bubble.appendChild(chartWrapper);
  chatBox.appendChild(bubble);
}

document.addEventListener('click', function (e) {
  if (e.target.classList.contains('zoom-btn') || e.target.closest('.chart-wrapper')) {
    const chart = e.target.closest('.chart-wrapper') || e.target.closest('.chat-chart-wrapper');
    const messageBlock = chart.closest('.message-block') || chart.closest('.chat-bubble');
    openChartModal(chart, messageBlock);
  }
});
