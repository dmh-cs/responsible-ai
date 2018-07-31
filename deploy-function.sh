#!/usr/bin/env bash
set -e

img="shiksb/ad_trust:$RANDOM"

docker build -t "$img" .
docker push "$img"

echo -e "\n\nRunning $img on docker...\n\n"
cat model/test_req.json | docker run -i $img ubuntu /bin/bash -c 'cat'

echo -e "\n\nRunning $img on cortex...\n\n"
cortex actions deploy --docker registry.cortex-develop.insights.ai:5000/$img at/ad_trust

