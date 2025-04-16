from django.contrib import admin
from .models import Teacher, User, Subject, SubjectDate, StudentAttendance

admin.site.register(Teacher)
admin.site.register(User)
admin.site.register(Subject)
admin.site.register(SubjectDate)
admin.site.register(StudentAttendance)