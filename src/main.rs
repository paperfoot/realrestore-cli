use clap::{Parser, Subcommand};
use comfy_table::{presets::UTF8_FULL, Table};
use owo_colors::OwoColorize;
use serde::Serialize;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const SKILL_CONTENT: &str = include_str!("../SKILL.md");

// ── CLI ──────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "realrestore", version = VERSION, about = "Agent-friendly image restoration CLI optimized for Apple Silicon")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Force JSON output even in a terminal
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Restore a degraded image
    Restore {
        /// Input image path
        input: PathBuf,

        /// Output image path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Restoration task
        #[arg(short, long, default_value = "auto")]
        task: String,

        /// Inference backend
        #[arg(short, long, default_value = "auto")]
        backend: String,

        /// Quantization level
        #[arg(short, long, default_value = "none")]
        quantize: String,

        /// Number of inference steps
        #[arg(long, default_value_t = 28)]
        steps: u32,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Quality preset (overrides steps)
        #[arg(long)]
        quality: Option<String>,

        /// Custom prompt (overrides task prompt)
        #[arg(long)]
        prompt: Option<String>,

        /// Enable tiling for high-resolution images
        #[arg(long)]
        tile: bool,

        /// Tile size in pixels (default: 512)
        #[arg(long, default_value_t = 512)]
        tile_size: u32,

        /// Tile overlap in pixels (default: 64)
        #[arg(long, default_value_t = 64)]
        tile_overlap: u32,
    },

    /// Remove invisible AI watermarks from an image
    WatermarkRemove {
        /// Input image path
        input: PathBuf,

        /// Output image path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Removal method
        #[arg(short, long, default_value = "ensemble")]
        method: String,
    },

    /// Run performance benchmarks
    Benchmark {
        /// Number of iterations
        #[arg(short, long, default_value_t = 3)]
        iterations: u32,

        /// Backends to benchmark (comma-separated)
        #[arg(short, long, default_value = "auto")]
        backends: String,

        /// Test image path (uses built-in if omitted)
        #[arg(long)]
        image: Option<PathBuf>,
    },

    /// Manage persistent inference daemon (keeps model in memory)
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },

    /// Print machine-readable capability manifest
    AgentInfo,

    /// Install skill file for AI agent platforms
    Skill {
        #[command(subcommand)]
        action: SkillAction,
    },

    /// Check for and apply updates
    Update,

    /// Set up Python environment and download models
    Setup {
        /// Model variant to download
        #[arg(long, default_value = "default")]
        model: String,
    },
}

#[derive(Subcommand)]
enum SkillAction {
    /// Install skill files to agent directories
    Install,
}

#[derive(Subcommand)]
enum DaemonAction {
    /// Start the inference daemon (keeps model in memory)
    Start {
        /// Inference backend
        #[arg(short, long, default_value = "auto")]
        backend: String,
        /// Quantization level
        #[arg(short, long, default_value = "none")]
        quantize: String,
    },
    /// Stop the running daemon
    Stop,
    /// Check daemon status
    Status,
}

// ── Output ───────────────────────────────────────────────────────────

#[derive(Serialize)]
struct JsonEnvelope<T: Serialize> {
    version: &'static str,
    status: &'static str,
    data: T,
}

#[derive(Serialize)]
struct JsonError {
    version: &'static str,
    status: &'static str,
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    code: String,
    message: String,
    suggestion: String,
}

fn use_json(cli: &Cli) -> bool {
    cli.json || !std::io::stdout().is_terminal()
}

fn print_success<T: Serialize>(cli: &Cli, data: T, human_msg: &str) {
    if use_json(cli) {
        let envelope = JsonEnvelope {
            version: "1",
            status: "success",
            data,
        };
        println!("{}", serde_json::to_string(&envelope).unwrap());
    } else {
        println!("{}", human_msg);
    }
}

fn print_error(cli: &Cli, code: &str, message: &str, suggestion: &str) -> ! {
    let err = JsonError {
        version: "1",
        status: "error",
        error: ErrorDetail {
            code: code.to_string(),
            message: message.to_string(),
            suggestion: suggestion.to_string(),
        },
    };
    eprintln!("{}", serde_json::to_string(&err).unwrap());
    let exit_code = match code {
        "transient" => 1,
        "config" => 2,
        "invalid_input" => 3,
        "rate_limited" => 4,
        _ => 1,
    };
    std::process::exit(exit_code);
}

// ── Python bridge ────────────────────────────────────────────────────

fn python_dir() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    let base = exe.parent().unwrap_or_else(|| std::path::Path::new("."));
    // In dev: look relative to repo root
    let candidates = [
        base.join("../python"),
        base.join("../../python"),
        PathBuf::from("python"),
    ];
    for c in &candidates {
        if c.join("realrestore_cli").exists() {
            return c.to_path_buf();
        }
    }
    PathBuf::from("python")
}

fn find_python() -> String {
    for cmd in ["python3.12", "python3", "python"] {
        if which::which(cmd).is_ok() {
            return cmd.to_string();
        }
    }
    "python3".to_string()
}

fn run_python(cli: &Cli, module: &str, args: &[&str]) -> Result<serde_json::Value, String> {
    let python = find_python();
    let py_dir = python_dir();

    let mut cmd = Command::new(&python);
    cmd.arg("-m")
        .arg(module)
        .args(args)
        .env("PYTHONPATH", &py_dir)
        .env("REALRESTORE_JSON_OUTPUT", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let start = Instant::now();
    let output = cmd.output().map_err(|e| format!("Failed to run Python: {e}"))?;
    let elapsed = start.elapsed();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Python process failed ({}ms): {}", elapsed.as_millis(), stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Find the last JSON line (skip progress output)
    let json_line = stdout
        .lines()
        .rev()
        .find(|l| l.starts_with('{'))
        .unwrap_or("{}");

    serde_json::from_str(json_line).map_err(|e| format!("Invalid JSON from Python: {e}"))
}

// ── Commands ─────────────────────────────────────────────────────────

fn cmd_restore(cli: &Cli, input: &PathBuf, output: &Option<PathBuf>, task: &str, backend: &str, quantize: &str, steps: u32, seed: u64, prompt: &Option<String>) {
    let input_str = input.to_string_lossy().to_string();
    if !input.exists() {
        print_error(cli, "invalid_input", &format!("File not found: {input_str}"), "Check the file path and try again.");
    }

    let out_path = output.clone().unwrap_or_else(|| {
        let stem = input.file_stem().unwrap_or_default().to_string_lossy();
        let ext = input.extension().unwrap_or_default().to_string_lossy();
        input.with_file_name(format!("{stem}_restored.{ext}"))
    });
    let out_str = out_path.to_string_lossy().to_string();

    if !use_json(cli) {
        println!("{} {}", "Restoring".green().bold(), &input_str);
        println!("  Task: {task}  Backend: {backend}  Steps: {steps}");
    }

    let prompt_str = prompt.as_deref().unwrap_or("");
    let steps_str = steps.to_string();
    let seed_str = seed.to_string();
    let arg_strs: Vec<&str> = vec![
        "--input", &input_str,
        "--output", &out_str,
        "--task", task,
        "--backend", backend,
        "--quantize", quantize,
        "--steps", &steps_str,
        "--seed", &seed_str,
        "--prompt", prompt_str,
    ];

    match run_python(cli, "realrestore_cli.engine", &arg_strs) {
        Ok(result) => {
            let human_msg = format!(
                "{} Saved to {}\n  Time: {}s  Peak memory: {}MB",
                "Done!".green().bold(),
                out_path.display(),
                result.get("elapsed_seconds").and_then(|v| v.as_f64()).unwrap_or(0.0),
                result.get("peak_memory_mb").and_then(|v| v.as_f64()).unwrap_or(0.0),
            );
            print_success(cli, result, &human_msg);
        }
        Err(e) => print_error(cli, "transient", &e, "Check GPU memory and try with --quantize int8 or fewer --steps."),
    }
}

fn cmd_watermark_remove(cli: &Cli, input: &PathBuf, output: &Option<PathBuf>, method: &str) {
    let input_str = input.to_string_lossy().to_string();
    if !input.exists() {
        print_error(cli, "invalid_input", &format!("File not found: {input_str}"), "Check the file path and try again.");
    }

    let out_path = output.clone().unwrap_or_else(|| {
        let stem = input.file_stem().unwrap_or_default().to_string_lossy();
        let ext = input.extension().unwrap_or_default().to_string_lossy();
        input.with_file_name(format!("{stem}_clean.{ext}"))
    });

    if !use_json(cli) {
        println!("{} {}", "Removing watermarks from".cyan().bold(), &input_str);
    }

    let out_str = out_path.to_string_lossy().to_string();
    let arg_strs: Vec<&str> = vec!["--input", &input_str, "--output", &out_str, "--method", method];

    match run_python(cli, "realrestore_cli.watermark.remover", &arg_strs) {
        Ok(result) => {
            let human_msg = format!("{} Cleaned image saved to {}", "Done!".green().bold(), out_path.display());
            print_success(cli, result, &human_msg);
        }
        Err(e) => print_error(cli, "transient", &e, "Try a different --method (spectral, diffusion, ensemble)."),
    }
}

fn cmd_benchmark(cli: &Cli, iterations: u32, backends: &str, image: &Option<PathBuf>) {
    if !use_json(cli) {
        println!("{}", "Running benchmarks...".yellow().bold());
    }

    let iter_str = iterations.to_string();
    let img_str = image.as_ref().map(|p| p.to_string_lossy().to_string());
    let mut arg_strs: Vec<&str> = vec!["--iterations", &iter_str, "--backends", backends];
    if let Some(ref s) = img_str {
        arg_strs.extend(["--image", s.as_str()]);
    }

    match run_python(cli, "realrestore_cli.benchmarks.runner", &arg_strs) {
        Ok(result) => {
            if use_json(cli) {
                print_success(cli, result, "");
            } else {
                // Pretty-print benchmark table
                let mut table = Table::new();
                table.load_preset(UTF8_FULL);
                table.set_header(vec!["Backend", "Avg Time (s)", "Peak Mem (MB)", "PSNR", "SSIM"]);

                if let Some(results) = result.get("results").and_then(|v| v.as_array()) {
                    for r in results {
                        table.add_row(vec![
                            r.get("backend").and_then(|v| v.as_str()).unwrap_or("-").to_string(),
                            format!("{:.2}", r.get("avg_time").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                            format!("{:.0}", r.get("peak_memory_mb").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                            format!("{:.2}", r.get("psnr").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                            format!("{:.4}", r.get("ssim").and_then(|v| v.as_f64()).unwrap_or(0.0)),
                        ]);
                    }
                }
                println!("{table}");
            }
        }
        Err(e) => print_error(cli, "transient", &e, "Ensure Python environment is set up: realrestore setup"),
    }
}

fn cmd_daemon(cli: &Cli, action: &DaemonAction) {
    let (subcommand, args) = match action {
        DaemonAction::Start { backend, quantize } => {
            ("start", vec!["--backend", backend.as_str(), "--quantize", quantize.as_str()])
        }
        DaemonAction::Stop => ("stop", vec![]),
        DaemonAction::Status => ("status", vec![]),
    };

    let mut all_args = vec![subcommand];
    all_args.extend(args);

    match run_python(cli, "realrestore_cli.daemon", &all_args) {
        Ok(result) => {
            let status = result.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
            let human_msg = match status {
                "started" => format!("{} Daemon started (PID: {})", "Running".green().bold(),
                    result.get("pid").and_then(|v| v.as_i64()).unwrap_or(0)),
                "already_running" => "Daemon is already running.".to_string(),
                "stopping" => "Daemon stopping...".to_string(),
                "not_running" => "Daemon is not running.".to_string(),
                _ => format!("Daemon status: {status}"),
            };
            print_success(cli, result, &human_msg);
        }
        Err(e) => print_error(cli, "transient", &e, "Check Python environment: realrestore setup"),
    }
}

fn cmd_agent_info(_cli: &Cli) {
    let info = serde_json::json!({
        "name": "realrestore",
        "version": VERSION,
        "description": "Image restoration CLI optimized for Apple Silicon. Restores degraded images (blur, noise, haze, rain, compression artifacts, low-light, moire, lens flare, reflections) and removes invisible AI watermarks.",
        "commands": {
            "restore": {
                "description": "Restore a degraded image to high quality",
                "usage": "realrestore restore <input> -o <output> [--task <task>] [--backend <backend>] [--quantize <level>]",
                "args": {
                    "input": "Path to degraded input image (required)",
                    "output": "Path for restored output image (default: <input>_restored.<ext>)",
                    "task": "Restoration type: auto|deblur|denoise|dehaze|derain|low_light|compression|moire|lens_flare|reflection (default: auto)",
                    "backend": "Inference backend: auto|mps|mlx|cpu (default: auto)",
                    "quantize": "Quantization: none|int8|int4 (default: none)",
                    "steps": "Denoising steps, lower=faster (default: 28)",
                    "seed": "Random seed for reproducibility (default: 42)",
                    "prompt": "Custom restoration prompt (overrides task default)"
                }
            },
            "watermark-remove": {
                "description": "Remove invisible AI watermarks from images",
                "usage": "realrestore watermark-remove <input> -o <output> [--method <method>]",
                "args": {
                    "input": "Path to watermarked image (required)",
                    "output": "Path for cleaned output (default: <input>_clean.<ext>)",
                    "method": "Removal method: spectral|diffusion|ensemble (default: ensemble)"
                }
            },
            "benchmark": {
                "description": "Run performance benchmarks across backends",
                "usage": "realrestore benchmark [--iterations N] [--backends mps,mlx]",
                "args": {
                    "iterations": "Number of benchmark iterations (default: 3)",
                    "backends": "Comma-separated backends to test (default: auto)",
                    "image": "Custom test image path"
                }
            },
            "setup": {
                "description": "Set up Python environment and download models",
                "usage": "realrestore setup [--model default]"
            },
            "agent-info": {
                "description": "Print this capability manifest as JSON"
            }
        },
        "exit_codes": {
            "0": "Success",
            "1": "Transient error (retry)",
            "2": "Configuration error (fix setup)",
            "3": "Bad input (fix arguments)",
            "4": "Rate limited (wait)"
        },
        "env_vars": {
            "REALRESTORE_MODEL_PATH": "Override default model directory",
            "REALRESTORE_BACKEND": "Override default backend (mps|mlx|cpu)",
            "HF_HOME": "HuggingFace cache directory"
        },
        "requirements": {
            "python": ">=3.12",
            "hardware": "Apple Silicon recommended (M1+, 32GB+ RAM)",
            "disk": "~25GB for model weights"
        }
    });

    // agent-info always outputs JSON
    println!("{}", serde_json::to_string_pretty(&info).unwrap());
}

fn cmd_skill_install(_cli: &Cli) {
    let dirs = [
        dirs::home_dir().map(|h| h.join(".claude/skills")),
        dirs::home_dir().map(|h| h.join(".codex/skills")),
        dirs::home_dir().map(|h| h.join(".gemini/skills")),
    ];

    for dir in dirs.iter().flatten() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("Warning: Could not create {}: {e}", dir.display());
            continue;
        }
        let path = dir.join("realrestore.md");
        match std::fs::write(&path, SKILL_CONTENT) {
            Ok(_) => println!("Installed skill to {}", path.display()),
            Err(e) => eprintln!("Warning: Could not write {}: {e}", path.display()),
        }
    }
}

fn cmd_setup(cli: &Cli, _model: &str) {
    if !use_json(cli) {
        println!("{}", "Setting up RealRestore CLI...".yellow().bold());
    }

    match run_python(cli, "realrestore_cli.setup", &[]) {
        Ok(result) => {
            let human_msg = format!("{} Environment ready.", "Setup complete!".green().bold());
            print_success(cli, result, &human_msg);
        }
        Err(e) => print_error(cli, "config", &e, "Ensure Python 3.12+ is installed and accessible."),
    }
}

fn cmd_update(cli: &Cli) {
    if !use_json(cli) {
        println!("{}", "Checking for updates...".yellow().bold());
    }

    match self_update::backends::github::Update::configure()
        .repo_owner("199-biotechnologies")
        .repo_name("realrestore-cli")
        .bin_name("realrestore")
        .current_version(VERSION)
        .build()
    {
        Ok(updater) => match updater.update() {
            Ok(status) => {
                let msg = if status.updated() {
                    format!("Updated to v{}", status.version())
                } else {
                    format!("Already at latest version (v{VERSION})")
                };
                print_success(cli, serde_json::json!({"version": VERSION, "updated": status.updated()}), &msg);
            }
            Err(e) => print_error(cli, "transient", &format!("Update check failed: {e}"), "Try again later or install manually."),
        },
        Err(e) => print_error(cli, "config", &format!("Update configuration error: {e}"), "Report this issue on GitHub."),
    }
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Restore { input, output, task, backend, quantize, steps, seed, quality, prompt, tile, tile_size, tile_overlap } => {
            // Quality presets override steps
            let effective_steps = match quality.as_deref() {
                Some("fast") => 8,
                Some("balanced") => 14,
                Some("high") => 28,
                _ => *steps,
            };
            cmd_restore(&cli, input, output, task, backend, quantize, effective_steps, *seed, prompt);
        }
        Commands::WatermarkRemove { input, output, method } => {
            cmd_watermark_remove(&cli, input, output, method);
        }
        Commands::Benchmark { iterations, backends, image } => {
            cmd_benchmark(&cli, *iterations, backends, image);
        }
        Commands::Daemon { action } => cmd_daemon(&cli, action),
        Commands::AgentInfo => cmd_agent_info(&cli),
        Commands::Skill { action } => match action {
            SkillAction::Install => cmd_skill_install(&cli),
        },
        Commands::Update => cmd_update(&cli),
        Commands::Setup { model } => cmd_setup(&cli, model),
    }
}
