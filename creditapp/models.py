from django.db import models


class CreditApplication(models.Model):
    applicant_name = models.CharField(max_length=255)
    email = models.EmailField()
    ssn_last4 = models.CharField(max_length=4)
    annual_income = models.DecimalField(max_digits=12, decimal_places=2)
    requested_amount = models.DecimalField(max_digits=12, decimal_places=2)
    employment_status = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20,
        default="pending",
        choices=[
            ("pending", "Pending"),
            ("approved", "Approved"),
            ("rejected", "Rejected"),
        ],
    )

    class Meta:
        db_table = "credit_applications"
        managed = False

    def __str__(self):
        return f"{self.applicant_name} — ${self.requested_amount:,.2f} ({self.status})"


class Transaction(models.Model):
    applicant = models.ForeignKey(
        CreditApplication, on_delete=models.CASCADE, related_name="transactions"
    )
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    merchant = models.CharField(max_length=255)
    category = models.CharField(max_length=100)
    transaction_time = models.DateTimeField()
    location_country = models.CharField(max_length=100)
    is_online = models.BooleanField(default=False)

    class Meta:
        db_table = "transactions"
        managed = False

    def __str__(self):
        return f"{self.merchant} — ${self.amount} ({self.transaction_time:%Y-%m-%d})"


class FraudResult(models.Model):
    application = models.ForeignKey(
        CreditApplication, on_delete=models.CASCADE, related_name="fraud_results"
    )
    rule_name = models.CharField(max_length=100)
    triggered = models.BooleanField(default=False)
    score = models.DecimalField(max_digits=5, decimal_places=2, null=True)
    details = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "fraud_results"
        managed = False

    def __str__(self):
        status = "TRIGGERED" if self.triggered else "clean"
        return f"{self.rule_name}: {status} (score: {self.score})"
