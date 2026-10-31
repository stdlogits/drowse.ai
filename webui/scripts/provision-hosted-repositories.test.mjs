import assert from "node:assert/strict";
import {
  parseArguments,
  provisionRepositories,
  repositoryPlan,
} from "./provision-hosted-repositories.mjs";

const runtimeLock = {
  models: [
    { convertedRepository: "stdlogits/model-a" },
    { convertedRepository: "stdlogits/model-b" },
  ],
};
const distributionLock = {
  catalogUrl: "https://huggingface.co/stdlogits/catalog/resolve/main/catalog.json",
  signatureUrl: "https://huggingface.co/stdlogits/catalog/resolve/main/catalog.sig.json",
};

assert.deepEqual(parseArguments([]), { create: false });
assert.deepEqual(parseArguments(["--create"]), { create: true });
assert.throws(() => parseArguments(["--unknown"]), /usage/);
assert.deepEqual(repositoryPlan(runtimeLock, distributionLock), {
  owner: "stdlogits",
  repositories: [
    "stdlogits/catalog",
    "stdlogits/model-a",
    "stdlogits/model-b",
    "stdlogits/model-a-instruments",
    "stdlogits/model-b-instruments",
  ],
});

const calls = [];
const existing = new Set(["stdlogits/catalog", "stdlogits/model-a"]);
const run = async (command, args) => {
  calls.push([command, args]);
  if (args[0] === "auth") return { stdout: '{"user":"stdlogits","orgs":[]}\n' };
  if (args[0] === "repos") {
    existing.add(args[2]);
    return { stdout: "created\n" };
  }
  const repository = args[2];
  if (!existing.has(repository)) throw Object.assign(new Error("missing"), { exitCode: 1 });
  return { stdout: '{"private":false,"gated":false}\n' };
};

assert.deepEqual(
  await provisionRepositories({ create: true, runtimeLock, distributionLock, run }),
  [
    { repository: "stdlogits/catalog", private: false, gated: false },
    { repository: "stdlogits/model-a", private: false, gated: false },
    { repository: "stdlogits/model-b", private: false, gated: false },
    { repository: "stdlogits/model-a-instruments", private: false, gated: false },
    { repository: "stdlogits/model-b-instruments", private: false, gated: false },
  ],
);
assert.equal(calls.filter(([, args]) => args[0] === "repos").length, 3);

await assert.rejects(
  provisionRepositories({
    create: true,
    runtimeLock,
    distributionLock,
    run: async (command, args) => args[0] === "auth"
      ? { stdout: '{"user":"someone-else","orgs":[]}\n' }
      : { stdout: '{"private":false,"gated":false}\n' },
  }),
  /cannot provision stdlogits repositories/,
);

console.log("hosted repository provisioning checks passed");
