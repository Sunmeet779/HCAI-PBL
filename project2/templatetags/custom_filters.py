# project2/templatetags/custom_filters.py
from django import template
from django.template.defaultfilters import stringfilter

register = template.Library()

@register.filter(name='subtract')
def subtract(value, arg):
    """Subtracts the arg from the value"""
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter(name='divide')
def divide(value, arg):
    """Divides the value by the arg"""
    try:
        return float(value) / float(arg)
    except (ValueError, ZeroDivisionError, TypeError):
        return 0

@register.filter(name='multiply')
def multiply(value, arg):
    """Multiplies the value by the arg"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter(name='percentage')
def percentage(value, decimals=2):
    """Converts a decimal to percentage"""
    try:
        return f"{float(value) * 100:.{int(decimals)}f}%"
    except (ValueError, TypeError):
        return "0%"

@register.filter(name='format_date')
@stringfilter
def format_date(value, fmt="%Y-%m-%d %H:%M"):
    """Formats a date string"""
    from datetime import datetime
    try:
        return datetime.strptime(value, "%Y%m%d_%H%M%S").strftime(fmt)
    except (ValueError, TypeError):
        return value

@register.filter(name='get_item')
def get_item(dictionary, key):
    """Gets an item from a dictionary"""
    return dictionary.get(key, '')