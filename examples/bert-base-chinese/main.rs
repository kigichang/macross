use anyhow::Result;
use candle_core::{IndexOp, Tensor};
use candle_nn::ops::softmax;
use macross::models::bert::BertForMaskedLM;
use macross::{AutoModel, AutoTokenizer, PretrainedModel};

fn main() -> Result<()> {
    let test_strs = vec!["巴黎是[MASK]国的首都。", "生活的真谛是[MASK]。"];
    let device = macross::device(false)?;

    let pretrained_model = PretrainedModel::with_files(
        "kigichang/fix-bert-base-chinese",
        "config.json",
        "fix-bert-base-chinese.safetensors",
    );

    let tokenizer = AutoTokenizer::from_pretrained("kigichang/fix-bert-base-chinese")
        .map_err(anyhow::Error::msg)?;

    let bert =
        BertForMaskedLM::from_pretrained(pretrained_model, candle_core::DType::F32, &device)?;

    let mask_id: u32 = tokenizer
        .token_to_id("[MASK]")
        .ok_or(anyhow::Error::msg("No [MASK] token"))?;

    for test_str in test_strs {
        let ids = tokenizer
            .encode(test_str, true)
            .map_err(anyhow::Error::msg)?;
        // println!("ids: {:?}", ids);
        let input_ids = Tensor::stack(&[Tensor::new(ids.get_ids(), &device)?], 0)?;
        let token_type_ids = Tensor::stack(&[Tensor::new(ids.get_type_ids(), &device)?], 0)?;
        let attention_mask = Tensor::stack(&[Tensor::new(ids.get_attention_mask(), &device)?], 0)?;
        let result = bert.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        let mask_idx = ids.get_ids().iter().position(|&x| x == mask_id).unwrap();
        let mask_token_logits = result.i((0, mask_idx, ..))?;
        let mask_token_probs = softmax(&mask_token_logits, 0)?;
        let mut top5_tokens: Vec<(usize, f32)> = mask_token_probs
            .to_vec1::<f32>()?
            .into_iter()
            .enumerate()
            .collect();
        top5_tokens.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top5_tokens = top5_tokens.into_iter().take(5).collect::<Vec<_>>();

        println!("Input: {}", test_str);
        for (idx, prob) in top5_tokens {
            println!(
                "{:?}: {:.3}",
                tokenizer.id_to_token(idx as u32).unwrap(),
                prob
            );
        }
        // {
        //     let (_, token_len, _) = result.dims3()?;
        //     let word_ids = ids.get_ids();
        //     //let words = test_str.chars().collect::<Vec<_>>();
        //     for i in 0..token_len {
        //         let token_logits = result.i((0, i, ..))?;
        //         let token_probs = softmax(&token_logits, 0)?;
        //         let mut top5_tokens: Vec<(usize, f32)> = token_probs
        //             .to_vec1::<f32>()?
        //             .into_iter()
        //             .enumerate()
        //             .collect();
        //         top5_tokens.sort_by(|a, b| b.1.total_cmp(&a.1));
        //         let top5_tokens = top5_tokens.into_iter().take(5).collect::<Vec<_>>();

        //         println!(
        //             "similar with ({i}){:?} are: ",
        //             tokenizer.id_to_token(word_ids[i]).unwrap()
        //         );
        //         for (idx, prob) in top5_tokens {
        //             println!(
        //                 "{:?}: {:.3}",
        //                 tokenizer.id_to_token(idx as u32).unwrap(),
        //                 prob
        //             );
        //         }
        //     }
        //     println!();
        // }
    }
    Ok(())
}
