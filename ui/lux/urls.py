from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^(?P<question_id>[1-8])/$', views.question, name='question'),
    url(r'^(?P<question_id>[1-8])/essay(?P<essay_id>[0-9]+)/$', views.essay, name='essay'),
    url(r'^$', views.index, name='index'),
]
