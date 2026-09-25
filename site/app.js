const sources = [
  {
    title: "AiR-D",
    body: "Chen, Jiang, Yang and Zhao, ECCV 2020. Eye tracking while people answer visual questions. The question is read before the image appears.",
  },
  {
    title: "GQA",
    body: "Hudson and Manning, CVPR 2019. The questions, short answers and scene-graph question types. AiR-D is a subset of the balanced validation split.",
  },
  {
    title: "Visual Genome",
    body: "Krishna et al., IJCV 2017. The photographs, CC BY 4.0. GQA image ids are Visual Genome image ids.",
  },
];

function barChart(target, rows, labelOf, valueOf) {
  const max = Math.max(...rows.map(valueOf));
  target.innerHTML = rows
    .map((row) => {
      const value = valueOf(row);
      const width = Math.max(2, (value / max) * 100);
      return `<div class="bar"><span>${labelOf(row)}</span><i style="width:${width}%"></i><b>${value}</b></div>`;
    })
    .join("");
}

function show(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("hidden", tab.id !== name));
  document.querySelectorAll("nav button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === name);
  });
}

document.querySelectorAll("nav button").forEach((button) => {
  button.addEventListener("click", () => show(button.dataset.tab));
});

document.getElementById("sources").innerHTML = sources
  .map((source) => `<article class="card"><strong style="font-size:18px">${source.title}</strong><span>${source.body}</span></article>`)
  .join("");

fetch("data.json")
  .then((response) => response.json())
  .then((data) => {
    const counts = data.counts;
    const tiles = [
      [counts.questions, "questions"],
      [counts.images, "images"],
      [counts.images_with_several_questions, "images with 2+ questions"],
      [counts.participants_per_question, "observers per question"],
    ];
    document.getElementById("counts").innerHTML = tiles
      .map(([value, label]) => `<div class="stat"><strong>${value.toLocaleString("en-GB")}</strong><span>${label}</span></div>`)
      .join("");

    document.getElementById("splits").innerHTML = Object.entries(data.splits)
      .map(
        ([name, split]) =>
          `<div class="stat"><strong>${split.questions}</strong><span>${name} · ${split.images} images</span></div>`,
      )
      .join("");

    const sizes = data.images;
    document.getElementById("image-sizes").textContent =
      `Photograph width runs from ${sizes.width_min} to ${sizes.width_max} pixels ` +
      `(median ${sizes.width_median}), height from ${sizes.height_min} to ${sizes.height_max} ` +
      `(median ${sizes.height_median}).`;

    const accuracy = data.human_accuracy;
    document.getElementById("accuracy-summary").textContent =
      `Mean human accuracy is ${(accuracy.mean * 100).toFixed(1)}% ` +
      `(standard deviation ${(accuracy.std * 100).toFixed(1)} points). ` +
      `${accuracy.all_correct} questions were answered correctly by all ten observers, ` +
      `${accuracy.mixed} were mixed, and ${accuracy.all_wrong} were missed by everyone. ` +
      `Accuracy only takes values in tenths, because exactly ten people saw each question.`;

    barChart(
      document.getElementById("accuracy-bars"),
      accuracy.histogram,
      (row) => `${Math.round(row.accuracy * 100)}%`,
      (row) => row.n,
    );
    barChart(
      document.getElementById("structural-bars"),
      data.question_types.structural,
      (row) => row[0],
      (row) => row[1],
    );
    barChart(
      document.getElementById("semantic-bars"),
      data.question_types.semantic,
      (row) => row[0],
      (row) => row[1],
    );

    const top = data.top_answers.map(([answer, n]) => `${answer} (${n})`).join(", ");
    document.getElementById("answers").textContent =
      `${data.counts.answer_vocabulary} distinct short answers. The most common are ${top}.`;

    const questionsOnImage = {};
    data.examples.forEach((example) => {
      (questionsOnImage[example.image_id] ||= []).push(example);
    });
    const repeated = Object.values(questionsOnImage).filter((group) => group.length > 1).length;
    if (repeated > 0) {
      const note = document.getElementById("gallery-note");
      note.textContent +=
        ` ${repeated} photographs appear twice. The gaze map is not a property of the image: it moves when the question changes.`;
    }

    document.getElementById("gallery").innerHTML = data.examples
      .map((example) => {
        const siblings = questionsOnImage[example.image_id].filter((other) => other.qid !== example.qid);
        const sibling = siblings.length
          ? ` The other question on this photograph in the gallery is “${siblings[0].question}”.`
          : "";
        const shared =
          example.n_questions_on_image > 1
            ? `${example.n_questions_on_image} questions use this image.`
            : "This image has one question.";
        return `
        <article class="example">
          <div class="pair">
            <figure>
              <img src="${example.image}" alt="Photograph for question ${example.qid}" />
              <figcaption>Image ${example.image_id}</figcaption>
            </figure>
            <figure>
              <img src="${example.gaze}" alt="Human gaze on the same photograph" />
              <figcaption>Gaze, seconds 1–3</figcaption>
            </figure>
          </div>
          <div class="text">
            <span class="tag">${example.note} · ${example.split}</span>
            <p><strong>${example.question}</strong></p>
            <p>Answer: ${example.answer}. Human accuracy ${Math.round(example.accuracy * 100)}%.
               Type ${example.structural} / ${example.semantic}.
               ${shared}${sibling}
            </p>
          </div>
        </article>`;
      })
      .join("");
  })
  .catch((error) => {
    document.getElementById("counts").textContent = `Could not load data.json (${error}).`;
  });
