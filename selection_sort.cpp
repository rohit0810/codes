#include<bits/stdc++.h>
using namespace std;

void selection_sort(int a[], int n){
	int min, temp;
	for(int i=0; i<n-1; i++){
		min=i;
		for(int j=i; j<n; j++){
			if(a[j]<a[min]){
				min=j;
			}
		}
		temp=a[min];
		a[min]=a[i];
		a[i]=temp;
	}
}

int main(){
	int a[]={12,53,63,74,23};
	int n=sizeof(a)/sizeof(a[0]);

	selection_sort(a,n);
	for(int i=0; i<n; i++){
		cout<<a[i]<<" ";
	}
}