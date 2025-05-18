from django import template

register = template.Library()
@register.filter(name='add_class')
def add_class(field, css):
    # Only add class if it's a form field
    if hasattr(field, 'as_widget'):
        return field.as_widget(attrs={"class": css})
    return field  # Return as-is if not a form field