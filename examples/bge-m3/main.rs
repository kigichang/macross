use anyhow::Result;
use candle_core::{IndexOp, Tensor};
use macross::{AutoModel, AutoTokenizer};

fn main() -> Result<()> {
    const MODLE_NAME: &str = "BAAI/bge-m3";

    let device = macross::device(false)?;
    let sentences = vec!["這是一個文本", "這是另一個文本"];
    let tokenizer = {
        let mut tokenizer =
            AutoTokenizer::from_pretrained(MODLE_NAME).map_err(anyhow::Error::msg)?;
        let params = tokenizers::PaddingParams::default();
        println!("padding: {:?}", params);
        let truncation = tokenizers::TruncationParams::default();
        println!("truncate: {:?}", truncation);
        let tokenizer = tokenizer.with_padding(Some(params));
        let tokenizer = tokenizer
            .with_truncation(Some(truncation))
            .map_err(anyhow::Error::msg)?;
        tokenizer.clone()
    };
    // println!("tokenizer: {:?}", tokenizer);

    let bert = macross::models::bert::BertModel::from_pretrained(
        (MODLE_NAME, true),
        candle_core::DType::F32,
        &device,
    )?;

    let encoded_input = tokenizer
        .encode_batch(sentences, true)
        .map_err(anyhow::Error::msg)?;

    let ids = encoded_input
        .iter()
        .map(|e| Tensor::new(e.get_ids(), &device))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let ids = Tensor::stack(&ids, 0)?;
    // println!("ids: {:?}", ids.to_vec2::<u32>()?);

    let type_ids = encoded_input
        .iter()
        .map(|e| Tensor::new(e.get_type_ids(), &device))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let type_ids = Tensor::stack(&type_ids, 0)?;
    // println!("type_ids: {:?}", type_ids.to_vec2::<u32>()?);

    let attention_mask = encoded_input
        .iter()
        .map(|e| Tensor::new(e.get_attention_mask(), &device))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let attention_mask = Tensor::stack(&attention_mask, 0)?;
    println!("attention_mask: {:?}", attention_mask.to_vec2::<u32>()?);

    let result = bert.forward(&ids, &type_ids, &attention_mask)?;
    let result = result.i((.., 0))?.contiguous()?;
    //println!("result: {:?}", result.i(1)?.to_vec1::<f32>()?);
    println!("result: {:?}", result.to_vec2::<f32>()?);
    //let result = result.mean(1)?;
    // let mean = result.mean(1)?;
    // let result = macross::normalize(&mean)?;
    //println!("result: {:?}", result.to_vec2::<f32>()?);
    Ok(())
}