package dev.cardrhyme.muscriptormobile

import android.graphics.drawable.Drawable
import android.os.Build
import android.widget.Spinner

/** Kotlin-friendly accessor; Spinner#getPopupBackground exists from API 23 onward. */
val Spinner.popupBackgroundDrawable: Drawable?
    get() = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) getPopupBackground() else null
