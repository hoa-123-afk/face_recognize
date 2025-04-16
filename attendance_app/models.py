from django.db import models

class Teacher(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)
    phone = models.CharField(max_length=20)
    date_create = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='teacher_images/', null=True, blank=True)

    def __str__(self):
        return self.name

class User(models.Model):  # hoặc Student
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)
    phone = models.CharField(max_length=20)
    date_create = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='student_images/', null=True, blank=True)

    def __str__(self):
        return self.name

class Subject(models.Model):
    name = models.CharField(max_length=100)
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE)
    time_start = models.TimeField()
    time_end = models.TimeField()
    date_start = models.DateField()
    date_end = models.DateField()

    def __str__(self):
        return self.name

class SubjectStudent(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    student = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.student.name} - {self.subject.name}"

class SubjectDate(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    current_date = models.DateField()
    status = models.BooleanField(default=True)  # True = Có học, False = Nghỉ

    def __str__(self):
        return f"{self.subject.name} - {self.current_date}"

class StudentAttendance(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE)
    subject_date = models.ForeignKey(SubjectDate, on_delete=models.CASCADE)
    status = models.BooleanField(default=False)  # True = Có mặt, False = Vắng

    def __str__(self):
        return f"{self.student.name} - {self.subject_date}"
