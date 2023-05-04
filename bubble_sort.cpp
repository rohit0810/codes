#include<bits/stdc++.h>
using namespace std;

void bubblesort(int a[], int n){
	int temp;
	for(int i=0; i<n; i++){
		for(int j=0; j<n-i-1; j++){
			if(a[j]>a[j+1]){
				temp=a[j];
				a[j]=a[j+1];
				a[j+1]=temp;
			}
		}
	}
}

int main(){
	int a[]={12,54,32,65,44,89,33};
	int n=sizeof(a)/sizeof(a[0]);

	bubblesort(a, n);
	for(int i=0; i<n; i++){
		cout<<a[i]<<" ";
	}
}