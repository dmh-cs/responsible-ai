from cortex_client import InputMessage, OutputMessage

import compas_demo

# the entry point of your model
def main(params):
    msg = InputMessage.from_params(params)

    # properties
    # model = msg.properties.get('model')
    # labels = msg.properties.get('labels')

    # input
    text = msg.payload.get('text')

    result = compas_demo.run()

    return OutputMessage.create().with_payload(
        {
            'predictions':  result
        }
    ).to_params()

