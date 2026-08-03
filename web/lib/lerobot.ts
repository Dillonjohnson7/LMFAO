// Parse a LeRobot v3.0 dataset folder picked in the browser (via an
// <input webkitdirectory>). We read meta/info.json plus the episodes/tasks
// parquet (hyparquet, pure JS), and map each episode to its camera video File
// and its [from_timestamp, to_timestamp) window inside that (possibly shared)
// mp4. Nothing is uploaded; everything stays local.

import { parquetReadObjects } from "hyparquet";

export interface LeRobotEpisode {
  index: number;
  length: number;
  task: string;
  /** camera key -> { file, from, to } window inside the packed video. */
  videos: Record<string, { file: File; from: number; to: number }>;
}

export interface LeRobotDataset {
  name: string;
  fps: number;
  robotType: string;
  cameras: string[];
  totalFrames: number;
  episodes: LeRobotEpisode[];
}

interface EpisodeRow {
  episode_index: number | bigint;
  length: number | bigint;
  tasks?: string[];
  [key: string]: unknown;
}

function asNumber(v: unknown): number {
  if (typeof v === "bigint") return Number(v);
  if (typeof v === "number") return v;
  return Number(v);
}

/** Index the FileList by dataset-relative path (strip the top-level folder). */
function indexByRelativePath(files: FileList | File[]): Map<string, File> {
  const map = new Map<string, File>();
  for (const f of Array.from(files)) {
    // webkitRelativePath is "<picked-folder>/<...path>"; store without the root.
    const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
    const parts = rel.split("/");
    map.set(parts.slice(1).join("/") || rel, f);
  }
  return map;
}

function formatPath(template: string, vars: Record<string, string | number>): string {
  // LeRobot templates look like "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4".
  return template.replace(/\{(\w+)(?::0(\d+)d)?\}/g, (_m, key: string, pad?: string) => {
    const v = vars[key];
    if (v === undefined) throw new Error(`missing template variable ${key}`);
    return pad ? String(v).padStart(Number(pad), "0") : String(v);
  });
}

async function readParquet(file: File): Promise<Record<string, unknown>[]> {
  const buf = await file.arrayBuffer();
  return parquetReadObjects({ file: buf });
}

export async function parseLeRobotFolder(files: FileList | File[]): Promise<LeRobotDataset> {
  const byPath = indexByRelativePath(files);

  const infoFile = byPath.get("meta/info.json");
  if (!infoFile) {
    throw new Error(
      "This folder doesn't look like a LeRobot dataset: meta/info.json not found. " +
        "Pick the dataset root (the folder containing meta/ and videos/)."
    );
  }
  const info = JSON.parse(await infoFile.text());
  if (!info.features || !info.video_path) {
    throw new Error("meta/info.json is missing features/video_path; unsupported dataset version.");
  }

  const cameras = Object.keys(info.features).filter((k) => info.features[k]?.dtype === "video");
  if (cameras.length === 0) throw new Error("No video features found in this dataset.");

  // Task index -> text (optional file; episodes rows usually carry tasks too).
  const tasksByIndex = new Map<number, string>();
  const tasksFile = byPath.get("meta/tasks.parquet");
  if (tasksFile) {
    try {
      for (const row of await readParquet(tasksFile)) {
        tasksByIndex.set(asNumber(row.task_index), String(row.task ?? ""));
      }
    } catch {
      // Tasks are cosmetic for previewing; ignore a malformed file.
    }
  }

  // Episode metadata can be sharded across several parquet files.
  const episodeRows: EpisodeRow[] = [];
  const episodePaths = [...byPath.keys()]
    .filter((p) => p.startsWith("meta/episodes/") && p.endsWith(".parquet"))
    .sort();
  if (episodePaths.length === 0) throw new Error("No meta/episodes/*.parquet files found.");
  for (const p of episodePaths) {
    episodeRows.push(...((await readParquet(byPath.get(p)!)) as EpisodeRow[]));
  }
  episodeRows.sort((a, b) => asNumber(a.episode_index) - asNumber(b.episode_index));

  const episodes: LeRobotEpisode[] = [];
  for (const row of episodeRows) {
    const index = asNumber(row.episode_index);
    const videos: LeRobotEpisode["videos"] = {};
    for (const cam of cameras) {
      const chunk = row[`videos/${cam}/chunk_index`];
      const fileIdx = row[`videos/${cam}/file_index`];
      if (chunk === undefined || fileIdx === undefined) continue;
      const rel = formatPath(info.video_path, {
        video_key: cam,
        chunk_index: asNumber(chunk),
        file_index: asNumber(fileIdx),
      });
      const file = byPath.get(rel);
      if (!file) continue; // video shard missing from the picked folder
      videos[cam] = {
        file,
        from: asNumber(row[`videos/${cam}/from_timestamp`] ?? 0),
        to: asNumber(row[`videos/${cam}/to_timestamp`] ?? 0),
      };
    }
    const taskText =
      (Array.isArray(row.tasks) && row.tasks.length > 0 && String(row.tasks[0])) ||
      tasksByIndex.get(0) ||
      "";
    episodes.push({ index, length: asNumber(row.length), task: taskText, videos });
  }

  const usable = episodes.filter((e) => Object.keys(e.videos).length > 0);
  if (usable.length === 0) {
    throw new Error(
      "Found episode metadata but none of the referenced video files. " +
        "Make sure the videos/ folder was included when picking the dataset."
    );
  }

  // Only offer cameras whose video files are actually present in the picked
  // folder (a partial local copy may carry a subset of the declared cameras).
  const availableCameras = cameras.filter((cam) => usable.some((e) => e.videos[cam]));

  return {
    name: infoFile.webkitRelativePath?.split("/")[0] || "dataset",
    fps: Number(info.fps ?? 30),
    robotType: String(info.robot_type ?? ""),
    cameras: availableCameras,
    totalFrames: Number(info.total_frames ?? 0),
    episodes: usable,
  };
}
