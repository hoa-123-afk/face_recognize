# Generated by Django 5.2 on 2025-04-16 14:52

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('attendance_app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='SubjectStudent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('student', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='attendance_app.user')),
                ('subject', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='attendance_app.subject')),
            ],
        ),
    ]
