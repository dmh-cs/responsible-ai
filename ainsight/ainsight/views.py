
from django.http import HttpResponse
from ainsight.compas_demo import run
from django.http import JsonResponse

def index(request):
    response = {}

    res = run()

    for r in res:
        response[r[0]] = r[1]

    return JsonResponse(response)