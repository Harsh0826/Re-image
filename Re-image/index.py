from django.http import HttpResponse
from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

def image_compressor(request):
    return render(request, 'image-compressor.html')

def online_image_converter(request):
    return render(request, 'online-image-converter.html')

def image_dehazer(request):
    return render(request, 'image-dehazer.html')

def image_advice(request):
    return render(request, 'image-advice.html')

def faq(request):
    return render(request, 'faq.html')

def about(request):
    return render(request, 'about.html')

def privacy(request):
    return render(request, 'privacy.html')

def privacy_policy(request):
    return render(request, 'privacy-policy.html')

def png_or_jpg_which_format_to_choose(request):
    return render(request, 'png-or-jpg-which-format-to-choose.html')

def how_to_optimize_images_for_the_web(request):
    return render(request, 'how-to-optimize-images-for-the-web.html')

def optimize_images_to_make_your_website_load_faster(request):
    return render(request, 'optimize-images-to-make-your-website-load-faster.html')

def resize_photos_without_losing_quality(request):
    return render(request, 'resize-photos-without-losing-quality.html')

def resize_scanned_documents(request):
    return render(request, 'resize-scanned-documents.html')

def how_to_resize_a_picture(request):
    return render(request, 'how-to-resize-a-picture.html')

def how_to_resize_a_photo(request):
    return render(request, 'how-to-resize-a-photo.html')

def how_can_i_resize_an_image(request):
    return render(request, 'how-can-i-resize-an-image.html')

def where_to_resize_pictures(request):
    return render(request, 'where-to-resize-pictures.html')

def how_to_edit_photo(request):
    return render(request, 'how-to-edit-photo.html')