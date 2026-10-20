// Gallery index: reads gallery/worlds.json and renders a card per world, each
// linking into the Spark viewer (index.html?src=<splats url>) for a walk-through.

interface WorldEntry {
  id: string;
  name: string;
  thumb: string; // relative url to a 2:1 thumbnail
  spz: string; // relative url to splats.spz (or .ply)
  prompt?: string;
}

const grid = document.getElementById("grid") as HTMLDivElement;
const empty = document.getElementById("empty") as HTMLDivElement;

async function main(): Promise<void> {
  let worlds: WorldEntry[] = [];
  try {
    const res = await fetch("gallery/worlds.json", { cache: "no-store" });
    if (res.ok) worlds = (await res.json()) as WorldEntry[];
  } catch {
    /* no manifest yet */
  }

  if (worlds.length === 0) {
    empty.hidden = false;
    return;
  }

  for (const w of worlds) {
    const card = document.createElement("a");
    card.className = "card";
    card.href = `index.html?src=${encodeURIComponent(w.spz)}`;

    const img = document.createElement("img");
    img.src = w.thumb;
    img.alt = w.name;
    img.loading = "lazy";

    const meta = document.createElement("div");
    meta.className = "meta";
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = w.name;
    const sub = document.createElement("div");
    sub.className = "sub";
    sub.textContent = w.prompt ?? "Walk through →";
    meta.append(name, sub);

    card.append(img, meta);
    grid.append(card);
  }
}

void main();
