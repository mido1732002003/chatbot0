// Script to generate the chatbot project file structure with empty files

const fs = require("fs");
const path = require("path");

// Define the project structure
const structure = {
  "README.md": "",
  "requirements.txt": "",
  core: {
    "__init__.py": "",
    "model.py": "",
    "attention.py": "",
    "embeddings.py": "",
    "generation.py": "",
    "layers.py": "",
  },
  training: {
    "__init__.py": "",
    "dataset.py": "",
    "trainer.py": "",
    "optimizer.py": "",
    "scheduler.py": "",
  },
  evaluation: {
    "__init__.py": "",
    "metrics.py": "",
    "perplexity.py": "",
    "evaluator.py": "",
  },
  alignment: {
    "__init__.py": "",
    "safety_filter.py": "",
    "dpo_trainer.py": "",
    "preference_dataset.py": "",
  },
  serving: {
    "__init__.py": "",
    "cli_chat.py": "",
    "api_server.py": "",
    "prompt_formatter.py": "",
  },
  configs: {
    "model.json": "",
    "train.json": "",
    "eval.json": "",
    "alignment.json": "",
    "serving.json": "",
    "tokenizer.json": "",
  },
  utils: {
    "__init__.py": "",
    "tokenizer.py": "",
    "logging_utils.py": "",
    "checkpoint.py": "",
    "seed.py": "",
    "text_utils.py": "",
  },
  data: {
    "train.jsonl": "",
    "val.jsonl": "",
    "preferences.jsonl": "",
    "safety_keywords.json": "",
  },
  scripts: {
    "__init__.py": "",
    "build_tokenizer.py": "",
    "run_sft.py": "",
    "run_dpo.py": "",
    "evaluate.py": "",
    "chat_cli.py": "",
    "serve_api.py": "",
    "smoke_test.py": "",
  },
  checkpoints: {
    ".gitkeep": "",
  },
};

// Helper function to create directories and files recursively
function createStructure(basePath, obj) {
  for (const [name, value] of Object.entries(obj)) {
    const fullPath = path.join(basePath, name);
    if (typeof value === "object") {
      // directory
      if (!fs.existsSync(fullPath)) {
        fs.mkdirSync(fullPath, { recursive: true });
        console.log("Created dir:", fullPath);
      }
      createStructure(fullPath, value);
    } else {
      // file
      if (!fs.existsSync(fullPath)) {
        fs.writeFileSync(fullPath, value);
        console.log("Created file:", fullPath);
      }
    }
  }
}

// Run in current directory
createStructure(process.cwd(), structure);
console.log("✅ Project structure created successfully!");
