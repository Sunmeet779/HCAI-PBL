// Add custom filters to Django template system
document.addEventListener('DOMContentLoaded', function() {
    if (typeof django !== 'undefined' && django.jQuery) {
        django.jQuery.fn.extend({
            split: function(separator) {
                return this.text().split(separator);
            }
        });
    }
});