const slides = [...document.querySelectorAll(".slide")];
const navButtons = [...document.querySelectorAll("[data-slide-target]")];
const currentSlide = document.getElementById("currentSlide");
let activeSlide = 0;

function showSlide(index) {
  activeSlide = (index + slides.length) % slides.length;
  slides.forEach((slide, slideIndex) => slide.classList.toggle("active", slideIndex === activeSlide));
  navButtons.forEach((button) => button.classList.toggle("active", Number(button.dataset.slideTarget) === activeSlide));
  if (currentSlide) currentSlide.textContent = String(activeSlide + 1);
}

navButtons.forEach((button) => {
  button.addEventListener("click", () => showSlide(Number(button.dataset.slideTarget)));
});

document.getElementById("prevSlide")?.addEventListener("click", () => showSlide(activeSlide - 1));
document.getElementById("nextSlide")?.addEventListener("click", () => showSlide(activeSlide + 1));

document.querySelectorAll("[data-figure-src]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-figure-src]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    document.getElementById("galleryImage").src = button.dataset.figureSrc;
    document.getElementById("galleryTitle").textContent = button.dataset.figureTitle;
  });
});

const predictionForm = document.getElementById("predictionForm");
predictionForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(predictionForm);
  const payload = {};

  formData.forEach((value, key) => {
    const numericValue = Number(value);
    payload[key] = value !== "" && Number.isFinite(numericValue) ? numericValue : value;
  });

  const label = document.getElementById("predictionLabel");
  const bars = document.getElementById("probabilityBars");
  label.textContent = "Memproses prediksi...";
  bars.innerHTML = "";

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();

  if (!response.ok) {
    label.textContent = result.error || "Prediksi gagal";
    return;
  }

  label.textContent = result.prediction;
  Object.entries(result.probabilities).forEach(([className, probability]) => {
    const percentage = Math.round(probability * 1000) / 10;
    const row = document.createElement("div");
    row.className = "probability-row";
    row.innerHTML = `
      <div class="probability-meta">
        <strong>${className}</strong>
        <span>${percentage.toFixed(1)}%</span>
      </div>
      <div class="probability-track">
        <div class="probability-fill" style="width: ${percentage}%"></div>
      </div>
    `;
    bars.appendChild(row);
  });
});

const retrainButton = document.getElementById("retrainButton");
retrainButton?.addEventListener("click", async () => {
  const output = document.getElementById("trainingOutput");
  retrainButton.disabled = true;
  output.textContent = "Training sedang berjalan. Mohon tunggu...";

  try {
    const response = await fetch("/api/retrain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rf_iterations: 10 }),
    });
    const result = await response.json();
    output.textContent = `${result.stdout || ""}\n${result.stderr || ""}`.trim();
    if (!response.ok) output.textContent = `Training gagal.\n\n${output.textContent}`;
  } catch (error) {
    output.textContent = `Training gagal: ${error.message}`;
  } finally {
    retrainButton.disabled = false;
  }
});

window.lucide && lucide.createIcons();
