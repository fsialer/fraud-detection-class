from django.urls import path

from . import views

app_name = "creditapp"

urlpatterns = [
    path("", views.index, name="index"),
    path("apply/", views.apply_credit, name="apply"),
    path("status/<int:application_id>/", views.application_status, name="status"),
    path("api/status/<int:application_id>/", views.application_status_api, name="status_api"),
    path("transactions/", views.transactions_feed, name="transactions"),
]
