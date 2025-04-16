from rest_framework import serializers
from .models import User, Teacher, Subject, SubjectDate, StudentAttendance, SubjectStudent

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'
class TeacherSerializer(serializers.ModelSerializer):
    class Meta:
        model = Teacher
        fields = '__all__'

class SubjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subject
        fields = '__all__'

class SubjectStudentSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubjectStudent
        fields = '__all__'

class SubjectDateSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubjectDate
        fields = '__all__'

class StudentAttendanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = StudentAttendance
        fields = '__all__'