#include<bits/stdc++.h>
using namespace std;

void insertionsort(int a[], int n){
	int temp;
	for(int i=0; i<n; i++){
		int j=i;
		while(j>0 && a[j-1]>a[j]){
			temp=a[j-1];
			a[j-1]=a[j];
			a[j]=temp;
			j--;
		}
	}
}

int main(){
	int a[]={12,85,75,95,64};
	int n=sizeof(a)/sizeof(a[0]);

	insertionsort(a, n);

	for(int i=0; i<n; i++){
		cout<<a[i]<<" ";
	}
}