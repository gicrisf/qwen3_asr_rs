use anyhow::{Context, Result};
use std::path::Path;
use std::time::Instant;

use qwen3_asr::inference::AsrInference;
use qwen3_asr::tensor::Device;

fn usage() -> ! {
    eprintln!("Qwen3-ASR libtorch benchmark");
    eprintln!();
    eprintln!("Usage: bench <model_path> <audio_file> [-n RUNS]");
    eprintln!();
    eprintln!("  -n N    Number of benchmark runs (default: 5)");
    std::process::exit(1);
}

struct Stats {
    min:  f64,
    mean: f64,
    max:  f64,
}

impl Stats {
    fn new(values: &[f64]) -> Self {
        let min  = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max  = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        Self { min, mean, max }
    }
}

fn main() -> Result<()> {
    // Silence tracing output during bench runs
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        usage();
    }

    let model_path = &args[1];
    let audio_file = &args[2];
    let mut runs = 5usize;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "-n" => {
                i += 1;
                runs = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!("error: -n requires a number");
                    std::process::exit(1);
                });
            }
            _ => usage(),
        }
        i += 1;
    }

    let model_dir = Path::new(model_path);
    if !model_dir.exists() {
        anyhow::bail!("model directory not found: {}", model_path);
    }
    if !Path::new(audio_file).exists() {
        anyhow::bail!("audio file not found: {}", audio_file);
    }

    // Detect audio duration via hound (WAV only; fallback to 0)
    let audio_ms = hound::WavReader::open(audio_file)
        .map(|r| {
            let spec = r.spec();
            let n_samples = r.into_samples::<i16>().count();
            n_samples as f64 / spec.sample_rate as f64 * 1000.0
        })
        .unwrap_or(0.0);

    #[cfg(feature = "tch-backend")]
    let device = if tch::Cuda::is_available() { Device::Gpu(0) } else { Device::Cpu };
    #[cfg(feature = "mlx")]
    let device = Device::Gpu(0);

    let n_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    eprintln!("system_info: n_cpus = {n_cpus}\n");

    eprint!("Loading model from {} ... ", model_path);
    let model = AsrInference::load(model_dir, device).context("Failed to load model")?;
    eprintln!("done");

    let audio_s = audio_ms / 1000.0;
    eprintln!("\nMode: full pipeline  |  {runs} run(s)  |  {audio_s:.1} s  [{audio_file}]\n");

    // Warmup
    eprint!("  warmup ... ");
    model.transcribe(audio_file, None)?;
    eprintln!("done");

    let mut total_ms_v = Vec::with_capacity(runs);

    for i in 0..runs {
        let t = Instant::now();
        let result = model.transcribe(audio_file, None)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let rt = ms / audio_ms;
        let n_words = result.text.split_whitespace().count();

        total_ms_v.push(ms);
        eprintln!(
            "  run {}/{runs}:  total={ms:6.0} ms  rt={rt:.2}x  words={n_words}",
            i + 1
        );
    }

    let tot = Stats::new(&total_ms_v);
    eprintln!();
    eprintln!("{:14}  {:>8}  {:>8}  {:>8}", "", "min", "mean", "max");
    eprintln!("{:14}  {:8.1}  {:8.1}  {:8.1}  ms",  "total",     tot.min,              tot.mean,              tot.max);
    eprintln!("{:14}  {:8.2}  {:8.2}  {:8.2}  x RT", "rt_factor", tot.min / audio_ms, tot.mean / audio_ms, tot.max / audio_ms);
    eprintln!();

    Ok(())
}
