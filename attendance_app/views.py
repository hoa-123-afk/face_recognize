from rest_framework import viewsets, status
from rest_framework.response import Response
from .models import User, Teacher, Subject, SubjectDate, StudentAttendance, SubjectStudent
from .serializers import UserSerializer, TeacherSerializer, SubjectSerializer, SubjectDateSerializer, StudentAttendanceSerializer, SubjectStudentSerializer

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
class TeacherViewSet(viewsets.ModelViewSet):
    queryset = Teacher.objects.all()
    serializer_class = TeacherSerializer

class SubjectViewSet(viewsets.ModelViewSet):
    queryset = Subject.objects.all()
    serializer_class = SubjectSerializer

class SubjectStudentViewSet(viewsets.ModelViewSet):
    queryset = SubjectStudent.objects.all()
    serializer_class = SubjectStudentSerializer

class SubjectDateViewSet(viewsets.ModelViewSet):
    queryset = SubjectDate.objects.all()
    serializer_class = SubjectDateSerializer

    def create(self, request, *args, **kwargs):
        # Tạo SubjectDate
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        subject_date = serializer.save()

        # Lấy subject từ subject_date
        subject = subject_date.subject

        # Lấy danh sách học sinh học subject này
        subject_students = SubjectStudent.objects.filter(subject=subject)

        # Tạo bản ghi StudentAttendance cho mỗi student
        attendance_list = []
        for ss in subject_students:
            attendance = StudentAttendance.objects.create(
                student=ss.student,
                subject_date=subject_date,
                status=False  # mặc định là vắng
            )
            attendance_list.append(attendance)

        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

class StudentAttendanceViewSet(viewsets.ModelViewSet):
    queryset = StudentAttendance.objects.all()
    serializer_class = StudentAttendanceSerializer