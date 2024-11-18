"""Utility functions for run Python files"""

def get_skip_layers(model, args):
    skip_layers = set()
    if args.skip_attention:
        for block in model.bert.encoder.layer:
            skip_layers.update({
                block.attention.self.query,
                block.attention.self.key,
                block.attention.self.value,
                block.attention.output,
            })
    if args.skip_first_block:
        first_block = model.bert.encoder.layer[0]
        skip_layers.update({
            first_block.attention.self.query,
            first_block.attention.self.key,
            first_block.attention.self.value,
            first_block.attention.output,
            first_block.intermediate.dense_act,
            first_block.output.dense
        })
    if args.skip_last_block:
        last_block = model.bert.encoder.layer[-1]
        skip_layers.update({
            last_block.attention.self.query,
            last_block.attention.self.key,
            last_block.attention.self.value,
            last_block.attention.output,
            last_block.intermediate.dense_act,
            last_block.output.dense
        })
    skip_layers.update({
                            model.bert.encoder.layer[0].attention.self.query,
                            model.bert.encoder.layer[0].attention.self.key,
                            model.bert.encoder.layer[0].attention.self.value,
    })
    try:
        skip_layers.update({
                                model.cls.predictions.decoder,
                                model.cls.seq_relationship
        })
    except:
        pass
    return skip_layers