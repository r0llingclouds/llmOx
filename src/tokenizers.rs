use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<String, i64>,
    unk_token_id: i64,  // ID for unknown words
}

impl Tokenizer {
    /// Creates a new tokenizer with a predefined vocabulary.
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        // Populate the vocabulary with some example words and their token IDs.
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);
        vocab.insert("rust".to_string(), 3);
        vocab.insert("programming".to_string(), 4);
        vocab.insert("language".to_string(), 5);

        Tokenizer {
            vocab,
            unk_token_id: 0,  // Typically, '0' or another number is reserved for unknown tokens.
        }
    }

    /// Encodes a text string to a vector of token IDs.
    ///
    /// This method splits the input text by spaces and converts each word into a token ID. Unknown words are mapped to the `unk_token_id`.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        text.split_whitespace()
            .map(|word| {
                self.vocab.get(word).cloned().unwrap_or(self.unk_token_id)
            })
            .collect()
    }

    /// Decodes a vector of token IDs back into a readable string.
    ///
    /// This is typically used for checking the outputs of models during testing or after inference.
    pub fn decode(&self, tokens: &[i64]) -> String {
        tokens.iter()
            .map(|&id| {
                self.vocab.iter().find(|&(_, &v)| v == id).map_or("<unk>".to_string(), |(k, _)| k.clone())
            })
            .collect::<Vec<String>>()
            .join(" ")
    }
}
