from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key, '')


@register.filter(name='uppercase')
def uppercase(value):
    """Converts a string to uppercase"""
    return value.upper()
