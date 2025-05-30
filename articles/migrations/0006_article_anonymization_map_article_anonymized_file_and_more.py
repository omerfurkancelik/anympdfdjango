# Generated by Django 5.0.4 on 2025-03-18 00:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('articles', '0005_referee_specialization'),
    ]

    operations = [
        migrations.AddField(
            model_name='article',
            name='anonymization_map',
            field=models.TextField(blank=True, help_text='JSON mapping of original text to anonymized text', null=True),
        ),
        migrations.AddField(
            model_name='article',
            name='anonymized_file',
            field=models.FileField(blank=True, null=True, upload_to='anonymized_articles/'),
        ),
        migrations.AddField(
            model_name='article',
            name='extracted_authors',
            field=models.TextField(blank=True, help_text='Automatically extracted author names', null=True),
        ),
        migrations.AddField(
            model_name='article',
            name='extracted_institutions',
            field=models.TextField(blank=True, help_text='Automatically extracted institutions', null=True),
        ),
        migrations.AddField(
            model_name='article',
            name='extracted_keywords',
            field=models.TextField(blank=True, help_text='Automatically extracted keywords', null=True),
        ),
        migrations.AddField(
            model_name='article',
            name='is_anonymized',
            field=models.BooleanField(default=False, help_text='Whether the article has been anonymized'),
        ),
    ]
