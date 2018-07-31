from cortex_client import InputMessage, OutputMessage

import run
import json

# the entry point of your model
def main(params):
    msg = InputMessage.from_params(params)

    # properties
    model = msg.properties.get('model')
    labels = msg.properties.get('labels')

    # input
    image = msg.payload.get('image')

    result = run.main(image, model, labels)

    return OutputMessage.create().with_payload(
        {
            'predictions':  result
        }
    ).to_params()

