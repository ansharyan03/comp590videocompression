use std::env;
use std::path::PathBuf;

use ffmpeg_sidecar::command::FfmpegCommand;
use workspace_root::get_workspace_root;

use std::fs::File;
use std::io::BufReader;
use std::io::{BufWriter, Write};

use bitbit::BitReader;
use bitbit::BitWriter;
use bitbit::MSB;

use toy_ac::decoder::Decoder;
use toy_ac::encoder::Encoder;
use toy_ac::symbol_model::VectorCountSymbolModel;

use ffmpeg_sidecar::event::StreamTypeSpecificData::Video;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Make sure ffmpeg is installed
    ffmpeg_sidecar::download::auto_download().unwrap();

    // Command line options
    // -verbose, -no_verbose                Default: -no_verbose
    // -report, -no_report                  Default: -report
    // -check_decode, -no_check_decode      Default: -no_check_decode
    // -skip_count n                        Default: -skip_count 0
    // -count n                             Default: -count 10
    // -in file_path                        Default: bourne.mp4 in data subdirectory of workplace
    // -out file_path                       Default: out.dat in data subdirectory of workplace

    // Set up default values of options
    let mut verbose = false;
    let mut report = true;
    let mut check_decode = false;
    let mut skip_count = 0;
    let mut count = 10;

    let mut data_folder_path = get_workspace_root();
    data_folder_path.push("data");

    let mut input_file_path = data_folder_path.join("bourne.mp4");
    let mut output_file_path = data_folder_path.join("out.dat");

    parse_args(
        &mut verbose,
        &mut report,
        &mut check_decode,
        &mut skip_count,
        &mut count,
        &mut input_file_path,
        &mut output_file_path,
    );

    // Run an FFmpeg command to decode video from input_file_path
    // Get output as grayscale (i.e., just the Y plane)
    let mut iter = FfmpegCommand::new() // <- Builder API like `std::process::Command`
        .input(input_file_path.to_str().unwrap())
        .format("rawvideo")
        .pix_fmt("gray8")
        .output("-")
        .spawn()? // <- Ordinary `std::process::Child`
        .iter()?; // <- Blocking iterator over logs and output

    let mut width = 0u32;
    let mut height = 0u32;

    let metadata = iter.collect_metadata()?;
    for i in 0..metadata.output_streams.len() {
        match &metadata.output_streams[i].type_specific_data {
            Video(vid_stream) => {
                width = vid_stream.width;
                height = vid_stream.height;

                if verbose {
                    println!(
                        "Found video stream at output stream index {} with dimensions {} x {}",
                        i, width, height
                    );
                }
                break;
            }
            _ => (),
        }
    }
    assert!(width != 0);
    assert!(height != 0);

    // Set up initial prior frame as uniform medium gray (y = 128)
    let mut prior_frame = vec![128u8; (width * height) as usize];

    let output_file = match File::create(&output_file_path) {
        Err(_) => panic!("Error opening output file"),
        Ok(f) => f,
    };

    // Setup bit writer and arithmetic encoder.
    let mut buf_writer = BufWriter::new(output_file);
    let mut bw = BitWriter::new(&mut buf_writer);
    let mut enc = Encoder::new();

    // 256 adaptive contexts, one for each possible predicted value
    let model_symbols: Vec<u8> = (0u8..=255).collect();
    let mut contexts: Vec<VectorCountSymbolModel<u8>> = (0..256)
        .map(|_| VectorCountSymbolModel::new(model_symbols.clone()))
        .collect();

    for frame in iter.filter_frames() {
        if frame.frame_num < skip_count {
            if verbose {
                println!("Skipping frame {}", frame.frame_num);
            }
            continue;
        }
        if frame.frame_num >= skip_count + count {
            break;
        }

        let current_frame: Vec<u8> = frame.data; // <- raw pixel y values
        let bits_start = enc.bits_written();

        // Process pixels in row major order.
        for r in 0..height {
            for c in 0..width {
                let idx = (r * width + c) as usize;

                // Spatio-temporal prediction: blend of prior frame (temporal) and
                // left/top neighbors in the current frame (spatial).
                let prediction = blend_neighbors(&prior_frame, &current_frame, width, r, c);

                // Residual: how wrong was the prediction? (modular, stays in u8)
                let residual = current_frame[idx].wrapping_sub(prediction);

                // Encode residual under the context for this prediction value
                enc.encode(&residual, &contexts[prediction as usize], &mut bw);
                contexts[prediction as usize].incr_count(&residual);
            }
        }

        if verbose {
            println!(
                "frame: {}, compressed size (bits): {}",
                frame.frame_num,
                enc.bits_written() - bits_start
            );
        }

        prior_frame = current_frame;
    }

    // Tie off arithmetic encoder and flush to file.
    enc.finish(&mut bw)?;
    bw.pad_to_byte()?;
    buf_writer.flush()?;

    // Decompress and check for correctness.
    if check_decode {
        let input_file = File::open(&output_file_path).expect("Error opening output file");
        let mut buf_reader = BufReader::new(input_file);
        let mut br: BitReader<_, MSB> = BitReader::new(&mut buf_reader);

        let dec_iter = FfmpegCommand::new() // <- Builder API like `std::process::Command`
            .input(input_file_path.to_str().unwrap())
            .format("rawvideo")
            .pix_fmt("gray8")
            .output("-")
            .spawn()? // <- Ordinary `std::process::Child`
            .iter()?; // <- Blocking iterator over logs and output

        let mut dec = Decoder::new();
        let mut dec_contexts: Vec<VectorCountSymbolModel<u8>> = (0..256)
            .map(|_| VectorCountSymbolModel::new(model_symbols.clone()))
            .collect();
        // Set up initial prior frame as uniform medium gray
        let mut prior_frame = vec![128u8; (width * height) as usize];

        'check: for frame in dec_iter.filter_frames() {
            if frame.frame_num >= skip_count + count {
                break;
            }
            if verbose {
                print!("Checking frame {} ... ", frame.frame_num);
            }

            let original_frame: Vec<u8> = frame.data; // <- raw pixel y values
            let mut reconstructed = vec![0u8; (width * height) as usize];

            // Process pixels in row major order.
            for r in 0..height {
                for c in 0..width {
                    let idx = (r * width + c) as usize;

                    // Must use the reconstructed frame for spatial neighbors (same as encoder)
                    let prediction = blend_neighbors(&prior_frame, &reconstructed, width, r, c);
                    let residual = dec
                        .decode(&dec_contexts[prediction as usize], &mut br)
                        .to_owned();
                    dec_contexts[prediction as usize].incr_count(&residual);

                    reconstructed[idx] = prediction.wrapping_add(residual);

                    if reconstructed[idx] != original_frame[idx] {
                        println!(
                            "error at ({}, {}): expected {}, got {}",
                            c, r, original_frame[idx], reconstructed[idx]
                        );
                        println!("Abandoning check.");
                        break 'check;
                    }
                }
            }
            if verbose {
                println!("correct.");
            }
            prior_frame = reconstructed;
        }
    }

    // Emit report
    if report {
        println!(
            "{} frames encoded, average size (bits): {}, compression ratio: {:.2}",
            count,
            enc.bits_written() / count as u64,
            (width * height * 8 * count) as f64 / enc.bits_written() as f64
        );
    }

    Ok(())
}

/// Blend the temporal neighbor (same pixel in prior frame) with the spatial
/// neighbors (left and top in the current frame) to form a prediction.
/// Falls back gracefully at row/column boundaries.
fn blend_neighbors(prior_frame: &[u8], current_frame: &[u8], width: u32, r: u32, c: u32) -> u8 {
    let idx   = (r * width + c) as usize;
    let prior = prior_frame[idx] as u32;
    let left  = if c > 0 { current_frame[idx - 1] as u32 } else { prior };
    let top   = if r > 0 { current_frame[((r - 1) * width + c) as usize] as u32 } else { prior };

    let predicted = match (r == 0, c == 0) {
        (true,  true)  => prior,
        (true,  false) => (prior + left + 1) / 2,
        (false, true)  => (prior + top  + 1) / 2,
        (false, false) => (prior + left + top) / 3,
    };

    predicted as u8
}

fn parse_args(
    verbose: &mut bool,
    report: &mut bool,
    check_decode: &mut bool,
    skip_count: &mut u32,
    count: &mut u32,
    input_file_path: &mut PathBuf,
    output_file_path: &mut PathBuf,
) {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-verbose" => *verbose = true,
            "-no_verbose" => *verbose = false,
            "-report" => *report = true,
            "-no_report" => *report = false,
            "-check_decode" => *check_decode = true,
            "-no_check_decode" => *check_decode = false,
            "-skip_count" => {
                *skip_count = args
                    .next()
                    .expect("Expected value after -skip_count")
                    .parse()
                    .unwrap()
            }
            "-count" => {
                *count = args
                    .next()
                    .expect("Expected value after -count")
                    .parse()
                    .unwrap()
            }
            "-in" => {
                *input_file_path =
                    PathBuf::from(args.next().expect("Expected path after -in"))
            }
            "-out" => {
                *output_file_path =
                    PathBuf::from(args.next().expect("Expected path after -out"))
            }
            _ => {}
        }
    }
}
