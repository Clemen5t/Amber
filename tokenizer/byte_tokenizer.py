class AmberByteTokenizer:
    """
    Amber Tokenizer v0.1

    IDs:
      0..255  = octets UTF-8
      256     = PAD
      257     = BOS
      258     = EOS
      259..4095 = réservés aux futures versions AmberTokenizer
    """

    PAD = 256
    BOS = 257
    EOS = 258

    # On conserve le vocabulaire 4096 d'Amber Seed.
    vocab_size = 4096
    active_vocab_size = 259

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False
    ):
        tokens = []

        if add_bos:
            tokens.append(self.BOS)

        tokens.extend(
            text.encode(
                "utf-8",
                errors="replace"
            )
        )

        if add_eos:
            tokens.append(self.EOS)

        return tokens

    def decode(self, tokens):
        byte_values = []

        for token in tokens:
            token = int(token)

            if 0 <= token <= 255:
                byte_values.append(token)

        return bytes(
            byte_values
        ).decode(
            "utf-8",
            errors="replace"
        )
