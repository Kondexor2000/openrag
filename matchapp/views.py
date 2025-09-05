from django.urls import reverse_lazy
from django.shortcuts import redirect
from django.views.generic import CreateView, UpdateView, DeleteView
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LogoutView
from django.contrib.auth.views import LoginView
from .forms import MatchForm
from django.http import Http404
from .models import Match, Score

# Create your views here.

class SignUpView(CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('match')

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('match')
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        response = super().form_valid(form)
        return response

class EditProfileView(LoginRequiredMixin, UpdateView):
    form_class = UserChangeForm
    success_url = reverse_lazy('match')

    def get_object(self):
        return self.request.user

    def form_valid(self, form):
        response = super().form_valid(form)
        return response

class DeleteAccountView(LoginRequiredMixin, DeleteView):
    success_url = reverse_lazy('login')

    def get_object(self, queryset=None):
        if self.request.user.is_authenticated:
            return self.request.user
        raise Http404("You are not logged in.")

    def form_valid(self, form):
        try:
            response = super().form_valid(form)
            return response

        except Exception as e:
            return redirect('delete_account')

class CustomLoginView(LoginView):
    redirect_authenticated_user = True

    def form_valid(self, form):
        remember_me = form.cleaned_data.get('remember_me')

        if remember_me:
            self.request.session.set_expiry(1209600 if remember_me else 0)  
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy('match')

class CustomLogoutView(LoginRequiredMixin, LogoutView):
    next_page = 'login'

class AddMatchView(LoginRequiredMixin, CreateView):
    form_class = MatchForm
    success_url = reverse_lazy("match")

class UpdateMatchView(LoginRequiredMixin, UpdateView):
    model = Match
    form_class = MatchForm
    success_url = reverse_lazy("match")

class DeleteMatchView(LoginRequiredMixin, DeleteView):
    model = Match
    success_url = reverse_lazy("match")

def score_request_user_view(request):
    user = request.user
    score = Score.objects.filter(player=user).order_by('score')

#   return render(request, template_name, {'score': score})