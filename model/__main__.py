from cortex_client import InputMessage, OutputMessage

import requests, json

# the entry point of your model
def main(params):
    msg = InputMessage.from_params(params)

    # properties
    # model = msg.properties.get('model')
    # labels = msg.properties.get('labels')

    # input
    text = msg.payload.get('text')

    result = json.loads(requests.request(method='GET', url='http://2c3ba607.ngrok.io/').text)

    return OutputMessage.create().with_payload(
        {
            'predictions':  result
        }
    ).to_params()

