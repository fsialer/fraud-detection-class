from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from .models import CreditApplication, Transaction


def index(request):
    applications = CreditApplication.objects.order_by("-created_at")[:20]
    return render(request, "creditapp/index.html", {"applications": applications})


def apply_credit(request):
    if request.method == "POST":
        app = CreditApplication.objects.create(
            applicant_name=request.POST["applicant_name"],
            email=request.POST["email"],
            ssn_last4=request.POST["ssn_last4"],
            annual_income=request.POST["annual_income"],
            requested_amount=request.POST["requested_amount"],
            employment_status=request.POST["employment_status"],
        )
        return redirect("creditapp:status", application_id=app.id)
    return render(request, "creditapp/apply.html")


def application_status(request, application_id):
    application = get_object_or_404(CreditApplication, id=application_id)
    fraud_results = application.fraud_results.order_by("rule_name")
    return render(
        request,
        "creditapp/status.html",
        {
            "application": application,
            "fraud_results": fraud_results,
        },
    )


def application_status_api(request, application_id):
    application = get_object_or_404(CreditApplication, id=application_id)
    fraud_results = application.fraud_results.order_by("rule_name")
    return JsonResponse(
        {
            "application_id": application.id,
            "applicant_name": application.applicant_name,
            "status": application.status,
            "fraud_results": [
                {
                    "rule_name": r.rule_name,
                    "triggered": r.triggered,
                    "score": float(r.score) if r.score else None,
                    "details": r.details,
                }
                for r in fraud_results
            ],
        }
    )


def transactions_feed(request):
    transactions = Transaction.objects.select_related("applicant").order_by(
        "-transaction_time"
    )[:50]
    return render(
        request, "creditapp/transactions.html", {"transactions": transactions}
    )
