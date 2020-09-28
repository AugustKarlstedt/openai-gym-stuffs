#include <stdbool.h>
#include <stdio.h>
#include <string.h>

// based on https://www.thegeekstuff.com/2013/06/buffer-overflow/

int main()
{
    char buffer[16];
    bool admin = false;

    printf("Enter the admin password: ");
    gets(buffer);

    if(strcmp(buffer, "password"))
    {
        printf("\nWrong Password");
    }
    else
    {
        printf("\nCorrect Password");
        admin = true;
    }

    if (admin)
    {
        printf("\nYou are now logged in.");
    }

    return 0;
}