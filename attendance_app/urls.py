from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserViewSet, TeacherViewSet, SubjectViewSet, SubjectDateViewSet, StudentAttendanceViewSet, SubjectStudentViewSet

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'teachers', TeacherViewSet)
router.register(r'subjects', SubjectViewSet)
router.register(r'subject-students', SubjectStudentViewSet)
router.register(r'subject-dates', SubjectDateViewSet)
router.register(r'attendance', StudentAttendanceViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
