# Generated by Django 3.2.13 on 2022-07-06 08:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('demo01', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ClusterImg',
            fields=[
                ('caseID', models.IntegerField(primary_key=True, serialize=False)),
                ('imgPath', models.CharField(max_length=255)),
            ],
        ),
    ]
