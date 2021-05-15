"""re_join_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from . import views

# 3rd video data
from django.conf import settings
from django.conf.urls.static import static

from . import index

urlpatterns = [
    # path('admin/', admin.site.urls),
    # path('', views.button),
    # path('output', views.output, name="script"),

    url(r'^admin/', admin.site.urls),
    url(r'^$', views.button),

    url(r'^index_method', views.index_method),
    url(r'^image_compressor_method', views.image_compressor_method),
    url(r'^online_image_converter_method', views.online_image_converter_method),
    url(r'^image_dehazer_method', views.image_dehazer_method),


    path('', index.index, name='index'),
    path('image-compressor', index.image_compressor, name='image_compressor'),
    path('online-image-converter', index.online_image_converter, name='online_image_converter'),
    path('image-dehazer', index.image_dehazer, name='image_dehazer'),
    path('image-advice', index.image_advice, name='image_advice'),
    path('faq', index.faq, name='faq'),
    path('about', index.about, name='about'),
    path('privacy', index.privacy, name='privacy'),
    path('privacy-policy', index.privacy_policy, name='privacy_policy'),
    path('png-or-jpg-which-format-to-choose', index.png_or_jpg_which_format_to_choose, name='png_or_jpg_which_format_to_choose'),
    path('how-to-optimize-images-for-the-web', index.how_to_optimize_images_for_the_web, name='how_to_optimize_images_for_the_web'),
    path('optimize-images-to-make-your-website-load-faster', index.optimize_images_to_make_your_website_load_faster, name='optimize_images_to_make_your_website_load_faster'),
    path('resize-photos-without-losing-quality', index.resize_photos_without_losing_quality, name='resize_photos_without_losing_quality'),
    path('resize-scanned-documents.html', index.resize_scanned_documents, name='resize_scanned_documents'),
    path('how-to-resize-a-picture.html', index.how_to_resize_a_picture, name='how_to_resize_a_picture'),
    path('how-to-resize-a-photo.html', index.how_to_resize_a_photo, name='how_to_resize_a_photo'),
    path('how-can-i-resize-an-image.html', index.how_can_i_resize_an_image, name='how_can_i_resize_an_image'),
    path('where-to-resize-pictures.html', index.where_to_resize_pictures, name='where_to_resize_pictures'),
    path('how-to-edit-photo.html', index.how_to_edit_photo, name='how_to_edit_photo'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)






