"""
URL configuration for match project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from matchapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', views.CustomLoginView.as_view(), name='login'),
    path('signup/', views.SignUpView.as_view(), name='signup'),
    path('logout/', views.CustomLogoutView.as_view(), name='logout'),
    path('edit-profile/', views.EditProfileView.as_view(), name='edit_profile'),
    path('delete-account/', views.DeleteAccountView.as_view(), name='delete_account'),
    path('/', views.score_request_user_view, name='match'),
    path('match/create', views.AddMatchView.as_view(), name='match_create'),
    path('match/<int:match_id>/update', views.UpdateMatchView.as_view(), name='match_update'),
    path('match/<int:match_id>/delete', views.DeleteMatchView.as_view(), name='match_delete'),
]
