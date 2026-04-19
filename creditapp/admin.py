from django.contrib import admin
from django.db.models import Count, Max, Q
from django.utils.html import format_html

from .models import CreditApplication, FraudResult, Transaction


class FraudResultInline(admin.TabularInline):
    model = FraudResult
    extra = 0
    readonly_fields = ("rule_name", "triggered_display", "score", "details", "created_at")
    fields = ("rule_name", "triggered_display", "score", "details", "created_at")
    ordering = ("rule_name",)

    def triggered_display(self, obj):
        if obj.triggered:
            return format_html('<span style="color: #CC0000; font-weight: 600;">TRIGGERED</span>')
        return format_html('<span style="color: #008A44;">clean</span>')

    triggered_display.short_description = "Status"

    def has_add_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.action(description="Mark as Approved")
def mark_approved(modeladmin, request, queryset):
    queryset.update(status="approved")


@admin.action(description="Mark as Rejected")
def mark_rejected(modeladmin, request, queryset):
    queryset.update(status="rejected")


@admin.action(description="Reset to Pending")
def reset_pending(modeladmin, request, queryset):
    queryset.update(status="pending")


@admin.register(CreditApplication)
class CreditApplicationAdmin(admin.ModelAdmin):
    list_display = (
        "applicant_name",
        "requested_amount",
        "annual_income",
        "status",
        "rules_triggered",
        "max_fraud_score",
        "created_at",
    )
    list_filter = ("status", "employment_status", "created_at")
    search_fields = ("applicant_name", "email")
    date_hierarchy = "created_at"
    actions = [mark_approved, mark_rejected, reset_pending]
    inlines = [FraudResultInline]
    list_per_page = 25

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .annotate(
                _rules_triggered=Count(
                    "fraud_results", filter=Q(fraud_results__triggered=True)
                ),
                _max_score=Max("fraud_results__score"),
            )
        )

    def rules_triggered(self, obj):
        count = obj._rules_triggered
        if count >= 2:
            return format_html('<span style="color: #CC0000; font-weight: 600;">{}</span>', count)
        if count == 1:
            return format_html('<span style="color: #B37400;">{}</span>', count)
        return format_html('<span style="color: #008A44;">0</span>')

    rules_triggered.short_description = "Rules Hit"
    rules_triggered.admin_order_field = "_rules_triggered"

    def max_fraud_score(self, obj):
        score = obj._max_score
        if score is None:
            return format_html('<span style="color: #999;">—</span>')
        if score > 50:
            color = "#CC0000"
        elif score > 25:
            color = "#B37400"
        else:
            color = "#008A44"
        return format_html('<span style="color: {}; font-weight: 600;">{:.1f}</span>', color, score)

    max_fraud_score.short_description = "Max Score"
    max_fraud_score.admin_order_field = "_max_score"


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ("applicant", "amount", "merchant", "category", "transaction_time")
    list_filter = ("category", "is_online")


@admin.register(FraudResult)
class FraudResultAdmin(admin.ModelAdmin):
    list_display = ("application_link", "rule_name", "triggered_display", "score", "details")
    list_filter = ("triggered", "rule_name")
    search_fields = ("application__applicant_name",)

    def triggered_display(self, obj):
        if obj.triggered:
            return format_html('<span style="color: #CC0000; font-weight: 600;">TRIGGERED</span>')
        return format_html('<span style="color: #008A44;">clean</span>')

    triggered_display.short_description = "Status"

    def application_link(self, obj):
        from django.urls import reverse

        url = reverse("admin:creditapp_creditapplication_change", args=[obj.application_id])
        return format_html('<a href="{}">{}</a>', url, obj.application)

    application_link.short_description = "Application"
